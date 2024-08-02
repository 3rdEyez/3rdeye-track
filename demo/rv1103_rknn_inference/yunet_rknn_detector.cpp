#include "yunet_rknn_detector.h"
#include "cv_utils.h"
#include "rga.h"
#include "im2d.h"
#include "RgaUtils.h"
#include "dma_alloc.h"

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

void YunetRKNN::preprocess(const cv::Mat &img, cv::Mat &in, letterbox_info &info)
{
    cv::Mat img_copy = img.clone();
    letterbox(img_copy, imgsz[0], imgsz[1], info);
    in = img_copy.clone();
}


std::vector<BBox> YunetRKNN::detect(const cv::Mat &img, float score_threshold, float nms_threshold)
{
    std::vector<BBox> bboxes;
    int ret = 0;

    // set input data. dst_rga_buf is for letterox output, and inputmem_rga_buf is for rknn input
    // you should copy dst_rga_buf to inputmem_rga_buf
    cv::Mat img_dst;
    letterbox_info info;
    preprocess(img, img_dst, info);
    int dst_height = img_dst.rows;
    int dst_width = img_dst.cols;
    int format = RK_FORMAT_BGR_888;
    rga_buffer_t dst_rga_buf, inputmem_rga_buf;
    char *dst_buf;
    int dst_fd = 0;
    rga_buffer_handle_t dst_handle, inputmem_handle;
    int dst_buf_size;
    
    dst_buf_size = dst_height * dst_width * get_bpp_from_format(format);
    ret = dma_buf_alloc(RV1106_CMA_HEAP_PATH, dst_buf_size, &dst_fd, (void **)&dst_buf);
    if (ret) {
        printf("dst_buf_size dma_buf_alloc failed\n");
        return bboxes;
    }
    memcpy(dst_buf, img_dst.data, dst_buf_size);
    dma_sync_cpu_to_device(dst_fd);
    dst_handle = importbuffer_fd(dst_fd, dst_buf_size);
    if (dst_handle == 0) {
        printf("importbuffer_fd failed\n");
        return bboxes;
    }
    dst_rga_buf = wrapbuffer_handle(dst_handle, dst_width, dst_height, format);

    inputmem_handle = importbuffer_fd(app_ctx.input_mems[0]->fd, dst_buf_size);
    if (inputmem_handle == 0) {
        printf("importbuffer_fd failed\n");
        return bboxes;
    }
    inputmem_rga_buf = wrapbuffer_handle(inputmem_handle, dst_width, dst_height, format);

    // copy dst_rga_buf to inputmem_rga_buf
    ret = imcheck(dst_rga_buf, inputmem_rga_buf, {}, {});
    if (IM_STATUS_NOERROR != ret) {
        printf("%d, check error! %s", __LINE__, imStrError((IM_STATUS)ret));
        return bboxes;
    }
    ret = imcopy(dst_rga_buf, inputmem_rga_buf);
    if (ret == IM_STATUS_SUCCESS) {
        printf("%s running success!\n");
    } else {
        printf("%s running failed\n");
        return bboxes;
    }
    /* rknn run */
    ret = rknn_run(app_ctx.rknn_ctx, nullptr);

    if (ret < 0) {
        printf("run error %d\n", ret);
        return bboxes;
    }

    // TODO: get output data
    rknn_tensor_mem **_outputs = (rknn_tensor_mem **)app_ctx.output_mems;
    // print app_ctx.output_mems[0][0~1000]
    for (int i = 0; i < 1600; i++) {
        printf("0x%X ", ((uint8_t *)_outputs[3]->virt_addr)[i]);
    }
    return bboxes;
}