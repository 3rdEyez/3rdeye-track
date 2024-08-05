#include "yunet_rknn_detector.h"
#include "cv_utils.h"
#include "det_utils.h"
#include "rga_buffer_helper.h"
#include <iostream>

void letterbox_rga(const cv::Mat& cv_src, cv::Mat &cv_dst, int new_width, int new_height, letterbox_info& info)
{
    int width = cv_src.cols;
    int height = cv_src.rows;
    int & offset_x = info.offset_x, & offset_y = info.offset_y;
    float & scale = info.scale;
    scale = std::min((float)new_width / (float)width, (float)new_height / (float)height);
    int new_unscaled_width = (int)(scale * (float)width);
    int new_unscaled_height = (int)(scale * (float)height);
    offset_x = (new_width - new_unscaled_width) / 2;
    offset_y = (new_height - new_unscaled_height) / 2;
    im_rect src_resize_rect, dst_rect;
    src_resize_rect = {0, 0, new_unscaled_width, new_unscaled_height};
    dst_rect = {offset_x, offset_y, new_unscaled_width, new_unscaled_height};

    rga_buffer_helper src(cv_src.cols, cv_src.rows, RK_FORMAT_BGR_888);
    rga_buffer_helper src_resize(new_unscaled_width, new_unscaled_height, RK_FORMAT_BGR_888);
    rga_buffer_helper dst(new_width, new_height, RK_FORMAT_BGR_888);
    src.load_from(cv_src.data, cv_src.cols * cv_src.rows * 3);
    imresize(src.get_rga_buf(), src_resize.get_rga_buf());
    improcess(src_resize.get_rga_buf(), dst.get_rga_buf(), {}, src_resize_rect, dst_rect, {}, IM_SYNC);
    cv_dst = cv::Mat(new_height, new_width, CV_8UC3);
    memcpy(cv_dst.data, dst.get_buf(), new_width * new_height * 3);
}

inline static int32_t __clip(float val, float min, float max)
{
    float f = val <= min ? min : (val >= max ? max : val);
    return f;
}
static int8_t qnt_f32_to_affine(float f32, int32_t zp, float scale)
{
    float dst_val = (f32 / scale) + zp;
    int8_t res = (int8_t)__clip(dst_val, -128, 127);
    return res;
}
static float deqnt_affine_to_f32(int8_t qnt, int32_t zp, float scale) { return ((float)qnt - (float)zp) * scale; }

inline float fast_exp(float x)
{
    union {
        uint32_t i;
        float f;
    } v{};
    v.i = (1 << 23) * (1.4426950409 * x + 126.93490512f);
    return v.f;
}

static std::vector<Box> box_decode(const std::vector<Box> &boxes, Vec3f grid, int stride)
{
    size_t height = grid.size();
    size_t width = grid[0].size();
    std::vector<Box> out;
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            float x = boxes[i * width + j].x * stride + grid[i][j][0];
            float y = boxes[i * width + j].y * stride + grid[i][j][1];
            float w = fast_exp(boxes[i * width + j].w) * stride;
            float h = fast_exp(boxes[i * width + j].h) * stride;
            out.push_back(Box{x, y, w, h});
        }
    }
    return out;
}

Box box_decode_at(Box box, Vec3f grid, int stride, int row, int col)
{
    size_t height = grid.size();
    size_t width = grid[0].size();
    Box out;
    float x = box.x * stride + grid[row][col][0];
    float y = box.y * stride + grid[row][col][1];
    float w = fast_exp(box.w) * stride;
    float h = fast_exp(box.h) * stride;
    out = Box{x, y, w, h};
    return out;
}

static std::vector<std::vector<float>> kps_decode(const std::vector<std::vector<float>> &kps, Vec3f grid, int stride)
{
    size_t height = grid.size();
    size_t width = grid[0].size();
    std::vector<std::vector<float>> out;
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            float x[5], y[5];
            for (int k = 0; k < 5; k++) {
                x[k] = kps[i * width + j][k * 2] * stride + grid[i][j][0];
                y[k] = kps[i * width + j][k * 2 + 1] * stride + grid[i][j][1];
            }
            out.push_back({x[0], y[0], x[1], y[1], x[2], y[2], x[3], y[3], x[4], y[4]});
        }
    }
    return out;
}

// 返回二维网格，辅助bbox解码。因为每个网格单元包含两个数(x, y)，所以最后使用Vec3f类型返回值
Vec3f meshgrid(int height, int width, int stride)
{
    Vec3f g(height, Vec2f(width));
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            g[i][j] = Vec1f{(float)j * stride, (float)i * stride};
        }
    }
    return g;
}

static void dump_tensor_attr(rknn_tensor_attr *attr)
{
    printf("  index=%d, name=%s, n_dims=%d, dims=[%d, %d, %d, %d], n_elems=%d, size=%d, fmt=%s, type=%s, qnt_type=%s, "
           "zp=%d, scale=%f\n",
           attr->index, attr->name, attr->n_dims, attr->dims[0], attr->dims[1], attr->dims[2], attr->dims[3],
           attr->n_elems, attr->size, get_format_string(attr->fmt), get_type_string(attr->type),
           get_qnt_type_string(attr->qnt_type), attr->zp, attr->scale);
}


YunetRKNN::YunetRKNN(const char *rknn_model)
{
    // Load RKNN Model
    int ret;
    int model_len = 0;
    rknn_context ctx = 0;

    ret = rknn_init(&ctx, (char *)rknn_model, 0, 0, NULL);

    if (ret < 0)
    {
        printf("rknn_init fail! ret=%d\n", ret);
        return;
    }
    
    // Get Model Input Output Number
    rknn_input_output_num io_num;
    ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret != RKNN_SUCC)
    {
        printf("rknn_query fail! ret=%d\n", ret);
        return;
    }
    printf("model input num: %d, output num: %d\n", io_num.n_input, io_num.n_output);

    // Get Model Input Info
    printf("input tensors:\n");
    rknn_tensor_attr input_attrs[io_num.n_input];
    memset(input_attrs, 0, sizeof(input_attrs));
    for (int i = 0; i < io_num.n_input; i++)
    {
        input_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_NATIVE_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC)
        {
            printf("rknn_query fail! ret=%d\n", ret);
            return;
        }
        dump_tensor_attr(&(input_attrs[i]));
    }

    // Get Model Output Info
    printf("output tensors:\n");
    rknn_tensor_attr output_attrs[io_num.n_output];
    memset(output_attrs, 0, sizeof(output_attrs));
    for (int i = 0; i < io_num.n_output; i++)
    {
        output_attrs[i].index = i;
        //When using the zero-copy API interface, query the native output tensor attribute
        ret = rknn_query(ctx, RKNN_QUERY_NATIVE_NHWC_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC)
        {
            printf("rknn_query fail! ret=%d\n", ret);
            return;
        }
        dump_tensor_attr(&(output_attrs[i]));
    }

    // default input type is int8 (normalize and quantize need compute in outside)
    // if set uint8, will fuse normalize and quantize to npu
    input_attrs[0].type = RKNN_TENSOR_UINT8;
    // default fmt is NHWC,1106 npu only support NHWC in zero copy mode
    input_attrs[0].fmt = RKNN_TENSOR_NHWC;
    printf("input_attrs[0].size_with_stride=%d\n", input_attrs[0].size_with_stride);
    app_ctx.input_mems[0] = rknn_create_mem(ctx, input_attrs[0].size_with_stride);

    // Set input tensor memory
    ret = rknn_set_io_mem(ctx, app_ctx.input_mems[0], &input_attrs[0]);
    if (ret < 0) {
        printf("input_mems rknn_set_io_mem fail! ret=%d\n", ret);
        return;
    }

    // Set output tensor memory
    for (uint32_t i = 0; i < io_num.n_output; ++i) {
        app_ctx.output_mems[i] = rknn_create_mem(ctx, output_attrs[i].size_with_stride);
        ret = rknn_set_io_mem(ctx, app_ctx.output_mems[i], &output_attrs[i]);
        if (ret < 0) {
            printf("output_mems rknn_set_io_mem fail! ret=%d\n", ret);
            return;
        }
    }

    // Set to context
    app_ctx.rknn_ctx = ctx;

    // TODO
    if (output_attrs[0].qnt_type == RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC)
    {
        app_ctx.is_quant = true;
    }
    else
    {
        app_ctx.is_quant = false;
    }

    app_ctx.io_num = io_num;
    app_ctx.input_attrs = (rknn_tensor_attr *)malloc(io_num.n_input * sizeof(rknn_tensor_attr));
    memcpy(app_ctx.input_attrs, input_attrs, io_num.n_input * sizeof(rknn_tensor_attr));
    app_ctx.output_attrs = (rknn_tensor_attr *)malloc(io_num.n_output * sizeof(rknn_tensor_attr));
    memcpy(app_ctx.output_attrs, output_attrs, io_num.n_output * sizeof(rknn_tensor_attr));

    if (input_attrs[0].fmt == RKNN_TENSOR_NCHW) 
    {
        printf("model is NCHW input fmt\n");
        app_ctx.model_channel = input_attrs[0].dims[1];
        app_ctx.model_height  = input_attrs[0].dims[2];
        app_ctx.model_width   = input_attrs[0].dims[3];
    } else 
    {
        printf("model is NHWC input fmt\n");
        app_ctx.model_height  = input_attrs[0].dims[1];
        app_ctx.model_width   = input_attrs[0].dims[2];
        app_ctx.model_channel = input_attrs[0].dims[3];
    } 

    printf("model input height=%d, width=%d, channel=%d\n",
           app_ctx.model_height, app_ctx.model_width, app_ctx.model_channel);

}

YunetRKNN::~YunetRKNN()
{
    if (app_ctx.input_attrs != NULL)
    {
        free(app_ctx.input_attrs);
        app_ctx.input_attrs = NULL;
    }
    if (app_ctx.output_attrs != NULL)
    {
        free(app_ctx.output_attrs);
        app_ctx.output_attrs = NULL;
    }
    for (int i = 0; i < app_ctx.io_num.n_input; i++) {
        if (app_ctx.input_mems[i] != NULL) {
            rknn_destroy_mem(app_ctx.rknn_ctx, app_ctx.input_mems[i]);
        }
    }
    for (int i = 0; i < app_ctx.io_num.n_output; i++) {
        if (app_ctx.output_mems[i] != NULL) {
            rknn_destroy_mem(app_ctx.rknn_ctx, app_ctx.output_mems[i]);
        }
    }
    if (app_ctx.rknn_ctx != 0)
    {
        rknn_destroy(app_ctx.rknn_ctx);
        app_ctx.rknn_ctx = 0;
    }
}


static int cvbgr888_rga_resize(cv::Mat &cv_src, cv::Mat &cv_dst, cv::Size dst_size)
{
    int ret = 0;
    int src_width = cv_src.cols;
    int src_height = cv_src.rows;
    int dst_width = dst_size.width;
    int dst_height = dst_size.height;
    int format = RK_FORMAT_BGR_888;
    rga_buffer_t src_rga_buf, dst_rga_buf;
    char *src_buf = NULL, *dst_buf = NULL;
    rga_buffer_handle_t src_handle, dst_handle;
    int src_buf_size, dst_buf_size;
    int src_fd = 0, dst_fd = 0;
    cv_dst.release();
    cv_dst = cv::Mat(dst_height, dst_width, CV_8UC3);

    memset(&src_rga_buf, 0, sizeof(rga_buffer_t));
    memset(&dst_rga_buf, 0, sizeof(rga_buffer_t));
    src_buf_size = src_width * src_height * get_bpp_from_format(format);
    dst_buf_size = dst_width * dst_height * get_bpp_from_format(format);
    ret = dma_buf_alloc(RV1106_CMA_HEAP_PATH, src_buf_size, &src_fd, (void **)&src_buf);
    if (ret) {
        std::cout << "src_buf_size dma_buf_alloc failed" << std::endl;
        goto out;    
    }
    ret = dma_buf_alloc(RV1106_CMA_HEAP_PATH, dst_buf_size, &dst_fd, (void **)&dst_buf);
    if (ret) {
        std::cout << "dst_buf_size dma_buf_alloc failed" << std::endl;
        goto out;    
    }
    memcpy(src_buf, cv_src.data, src_buf_size);
    dma_sync_cpu_to_device(src_fd);
    src_handle = importbuffer_fd(src_fd, src_buf_size);
    if (src_handle == 0) {
        std::cout << "importbuffer_fd failed" << std::endl;
        goto out;    
    }
    src_rga_buf = wrapbuffer_handle(src_handle, src_width, src_height, format);

    dst_handle = importbuffer_fd(dst_fd, dst_buf_size);
    if (dst_handle == 0) {
        std::cout << "importbuffer_fd failed" << std::endl;
        goto out;    
    }
    dst_rga_buf = wrapbuffer_handle(dst_handle, dst_width, dst_height, format);
    ret = imcheck(src_rga_buf, dst_rga_buf, {}, {});
    if (IM_STATUS_NOERROR != ret) {
        printf("%d, check error! %s", __LINE__, imStrError((IM_STATUS)ret));
        return -1;
    }
    ret = imresize(src_rga_buf, dst_rga_buf);
    if (ret == IM_STATUS_SUCCESS) {
        // printf("%s running success!\n");
    } else {
        printf("%s running failed\n");
        return -1;
    }

    memcpy(cv_dst.data, dst_buf, dst_buf_size);
    dma_sync_device_to_cpu(dst_fd);

    releasebuffer_handle(src_handle);
    releasebuffer_handle(dst_handle);
    dma_buf_free(src_buf_size, &src_fd, src_buf);
    dma_buf_free(dst_buf_size, &dst_fd, dst_buf);

    return 0;
out:
    if (src_buf != NULL)
        dma_buf_free(src_buf_size, &src_fd, src_buf);
    if (dst_buf != NULL)
        dma_buf_free(dst_buf_size, &dst_fd, dst_buf);
    return -1;
}

void YunetRKNN::preprocess(const cv::Mat &img, cv::Mat &in, letterbox_info &info)
{
    letterbox_rga(img, in, imgsz[0], imgsz[1], info);
}


static Vec3f grid_40x40 = meshgrid(40, 40, 8);
static Vec3f grid_20x20 = meshgrid(20, 20, 16);
static Vec3f grid_10x10 = meshgrid(10, 10, 32);
std::vector<BBox> YunetRKNN::detect(const cv::Mat &img, float score_threshold, float nms_threshold)
{
    std::vector<BBox> valid_result;
    std::vector<float> scores_keep;
    std::vector<Box> boxes, boxes_keep;
    std::vector<std::vector<float>> kps, kps_keep;
    int ret = 0;
    // set input data. dst_rga_buf is for letterox output, and inputmem_rga_buf is for rknn input
    // you should copy dst_rga_buf to inputmem_rga_buf
    rknn_tensor_mem **_outputs = (rknn_tensor_mem **)app_ctx.output_mems;
    letterbox_info info;
    cv::Mat img_dst;
    preprocess(img, img_dst, info);
    // print info
    // printf("letterbox info: x=%d, y=%d, s=%f\n", info.offset_x, info.offset_y, info.scale);
    int dst_height = img_dst.rows;
    int dst_width = img_dst.cols;
    int format = RK_FORMAT_BGR_888;
    rga_buffer_t dst_rga_buf, inputmem_rga_buf;
    char *dst_buf;
    int dst_fd = 0;
    rga_buffer_handle_t dst_handle = 0, inputmem_handle;
    int dst_buf_size;
    std::vector<float> out_tensor_scale_list;
    std::vector<int> out_tensor_zero_point_list;


    // alloc dma buffer for dst_rga_buf
    dst_buf_size = dst_height * dst_width * get_bpp_from_format(format);
    ret = dma_buf_alloc(RV1106_CMA_HEAP_PATH, dst_buf_size, &dst_fd, (void **)&dst_buf);
    if (ret) {
        printf("dst_buf_size dma_buf_alloc failed\n");
        goto out;
    }
    // copy cv::Mat to dma buffer
    memcpy(dst_buf, img_dst.data, dst_buf_size);
    
    dma_sync_cpu_to_device(dst_fd);

    // create rga buffer_handle
    dst_handle = importbuffer_fd(dst_fd, dst_buf_size);
    if (dst_handle == 0) {
        printf("importbuffer_fd failed\n");
        goto out;
    }
    dst_rga_buf = wrapbuffer_handle(dst_handle, dst_width, dst_height, format);

    inputmem_handle = importbuffer_fd(app_ctx.input_mems[0]->fd, dst_buf_size);
    if (inputmem_handle == 0) {
        printf("importbuffer_fd failed\n");
        goto out;
    }
    inputmem_rga_buf = wrapbuffer_handle(inputmem_handle, dst_width, dst_height, format);

    // copy dst_rga_buf to inputmem_rga_buf
    ret = imcheck(dst_rga_buf, inputmem_rga_buf, {}, {});
    if (IM_STATUS_NOERROR != ret) {
        printf("%d, check error! %s", __LINE__, imStrError((IM_STATUS)ret));
        goto out;
    }

    /* set up rknn input buffer */
    ret = imcopy(dst_rga_buf, inputmem_rga_buf);
    if (ret == IM_STATUS_SUCCESS) {
        // printf("%s running success!\n");
    } else {
        printf("%s running failed\n");
        goto out;
    }
    /* rknn run */
    ret = rknn_run(app_ctx.rknn_ctx, nullptr);
    if (ret < 0) {
        printf("run error %d\n", ret);
        goto out;
    }
    /* get zero points and scales */
    for (int i = 0; i < app_ctx.io_num.n_output; i++) {
        out_tensor_scale_list.push_back(app_ctx.output_attrs[i].scale);
        out_tensor_zero_point_list.push_back(app_ctx.output_attrs[i].zp);
    }
    // box_stride_8 keypoints_stride_8 meshgrid_40X40
    for (int i = 0; i < 1600; i++) {
        int8_t *pb = (int8_t *)_outputs[6]->virt_addr;
        int8_t *pk = (int8_t *)_outputs[9]->virt_addr;
        int8_t *box_row = pb + i * 4;
        boxes.push_back({
            deqnt_affine_to_f32(box_row[0], out_tensor_zero_point_list[6], out_tensor_scale_list[6]),
            deqnt_affine_to_f32(box_row[1], out_tensor_zero_point_list[6], out_tensor_scale_list[6]),
            deqnt_affine_to_f32(box_row[2], out_tensor_zero_point_list[6], out_tensor_scale_list[6]),
            deqnt_affine_to_f32(box_row[3], out_tensor_zero_point_list[6], out_tensor_scale_list[6])
        });
    }
    // boxes = box_decode(boxes, grid_40x40, 8);
    for (int i = 0; i < 1600; i++) {
        float score = deqnt_affine_to_f32(*((int8_t *)_outputs[0]->virt_addr + i), out_tensor_zero_point_list[0], out_tensor_scale_list[0]);
        score = score * deqnt_affine_to_f32(*((int8_t *)_outputs[3]->virt_addr + i), out_tensor_zero_point_list[3], out_tensor_scale_list[3]);
        score = sqrtf(score);
        if (score > score_threshold) { 
            scores_keep.push_back(score);
            boxes_keep.push_back(box_decode_at(boxes[i], grid_40x40, 8, i / 40, i % 40));
        } 
    }
    boxes.clear();
    kps.clear();
    
    // box_stride_16 keypoints_stride_16 meshgrid_20X20
    for (int i = 0; i < 400; i++) {
        int8_t *pb = (int8_t *)_outputs[7]->virt_addr;
        int8_t *pk = (int8_t *)_outputs[10]->virt_addr;
        int8_t *box_row = pb + i * 4;
        boxes.push_back({
            (float)(box_row[0] - out_tensor_zero_point_list[7]) * out_tensor_scale_list[7],
            (float)(box_row[1] - out_tensor_zero_point_list[7]) * out_tensor_scale_list[7],
            (float)(box_row[2] - out_tensor_zero_point_list[7]) * out_tensor_scale_list[7],
            (float)(box_row[3] - out_tensor_zero_point_list[7]) * out_tensor_scale_list[7]
        });
    }
    // boxes = box_decode(boxes, grid_20x20, 16);
    for (int i = 0; i < 400; i++) {
        float score = deqnt_affine_to_f32(*((int8_t *)_outputs[1]->virt_addr + i), out_tensor_zero_point_list[1], out_tensor_scale_list[1]);
        score = score * deqnt_affine_to_f32(*((int8_t *)_outputs[4]->virt_addr + i), out_tensor_zero_point_list[4], out_tensor_scale_list[4]);
        score = sqrtf(score);
        if (score > score_threshold) {
            scores_keep.push_back(score);
            boxes_keep.push_back(box_decode_at(boxes[i], grid_20x20, 16, i / 20, i % 20));
        }
    }
    boxes.clear();
    kps.clear();

    // box_stride_32 keypoints_stride_32 meshgrid_10X10 
    for (int i = 0; i < 100; i++) {
        int8_t *pb = (int8_t *)_outputs[8]->virt_addr;
        int8_t *pk = (int8_t *)_outputs[11]->virt_addr;
        int8_t *box_row = pb + i * 4;
        boxes.push_back({
            (float)(box_row[0] - out_tensor_zero_point_list[8]) * out_tensor_scale_list[8],
            (float)(box_row[1] - out_tensor_zero_point_list[8]) * out_tensor_scale_list[8],
            (float)(box_row[2] - out_tensor_zero_point_list[8]) * out_tensor_scale_list[8],
            (float)(box_row[3] - out_tensor_zero_point_list[8]) * out_tensor_scale_list[8]
        });
    }
    // boxes = box_decode(boxes, grid_10x10, 32);
    for (int i = 0; i < 100; i++) {
        float score = deqnt_affine_to_f32(*((int8_t *)_outputs[2]->virt_addr + i), out_tensor_zero_point_list[2], out_tensor_scale_list[2]);
        score = score * deqnt_affine_to_f32(*((int8_t *)_outputs[5]->virt_addr + i), out_tensor_zero_point_list[5], out_tensor_scale_list[5]);
        score = sqrtf(score);
        if (score > score_threshold) {
            scores_keep.push_back(score);
            boxes_keep.push_back(box_decode_at(boxes[i], grid_10x10, 32, i / 10, i % 10));
        }
    }
    
    for (int i = 0; i < boxes_keep.size(); i++) {
        auto &p = boxes_keep[i];
        valid_result.push_back({p.x, p.y, p.w, p.h, scores_keep[i], 0});
    }

    for (auto &box : valid_result) {
        box.x = (box.x - info.offset_x) / (info.scale + 0.001f);
        box.y = (box.y - info.offset_y) / (info.scale + 0.001f);
        box.w = box.w / (info.scale + 0.001f);
        box.h = box.h / (info.scale + 0.001f);
    }

    valid_result = nms(valid_result, nms_threshold);
    for (auto &box : valid_result) {
        printf("box: %f %f %f %f %f\n", box.x, box.y, box.w, box.h, box.score);
    }

out:
    if (dst_handle) {
        releasebuffer_handle(dst_handle);
    }
    if (releasebuffer_handle(inputmem_handle)) {
        releasebuffer_handle(inputmem_handle);
    }
    if (dst_buf != NULL)
        dma_buf_free(dst_buf_size, &dst_fd, dst_buf);
 
    return valid_result;
}