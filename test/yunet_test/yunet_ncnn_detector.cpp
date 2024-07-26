#include "yunet_ncnn_detector.h"
#include <cmath>

YunetNCNN::YunetNCNN(const char* param, const char* bin, bool useGPU)
{
    this->net = new ncnn::Net();
#if NCNN_USE_GPU
    this->hasGPU = ncnn::get_gpu_count() > 0;
#endif
    this->net->opt.use_vulkan_compute = this->hasGPU && useGPU;
    this->net->load_param(param);
    this->net->load_model(bin);
}

YunetNCNN::~YunetNCNN()
{
    delete this->net;
}


void YunetNCNN::preprocess(const cv::Mat &img, ncnn::Mat &in, letterbox_info &info)
{
    cv::Mat img_copy = img.clone();
    letterbox(img_copy, imgsz[0], imgsz[1], info);
    int img_w = img_copy.cols;
    int img_h = img_copy.rows;
    in = ncnn::Mat::from_pixels(img_copy.data, ncnn::Mat::PIXEL_BGR, img_w, img_h);
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

static std::vector<Box> box_decode(const std::vector<Box> &boxes, Vec3f grid, int stride)
{
    size_t height = grid.size();
    size_t width = grid[0].size();
    std::vector<Box> out;
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            float x = boxes[i * width + j].x * stride + grid[i][j][0];
            float y = boxes[i * width + j].y * stride + grid[i][j][1];
            float w = expf(boxes[i * width + j].w) * stride;
            float h = expf(boxes[i * width + j].h) * stride;
            out.push_back(Box{x, y, w, h});
        }
    }
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

std::vector<BBox> YunetNCNN::detect(const cv::Mat &img, float score_threshold, float nms_threshold)
{
    ncnn::Mat input;
    letterbox_info info;
    int cnt = 0;
    std::vector<BBox> valid_result;
    std::vector<float> scores_keep;
    std::vector<Box> boxes, boxes_keep;
    std::vector<std::vector<float>> kps, kps_keep;
    Vec3f grid_40x40 = meshgrid(40, 40, 8);
    Vec3f grid_20x20 = meshgrid(20, 20, 16);
    Vec3f grid_10x10 = meshgrid(10, 10, 32);
    std::vector<std::string> output_labels = {
        "cls_8", "cls_16", "cls_32", "obj_8", "obj_16", "obj_32", "bbox_8", "bbox_16", "bbox_32", "kps_8", "kps_16", "kps_32"};
    std::vector<ncnn::Mat> outs(output_labels.size());
    this->preprocess(img, input, info);
    auto ex = this->net->create_extractor();
    ex.input("input", input);
    for (int i = 0; i < output_labels.size(); i++) {
        ex.extract(output_labels[i].c_str(), outs[i]);
    }

    // box_stride_8 keypoints_stride_8 meshgrid_40X40
    for (int i = 0; i < 1600; i++) {
        float *box_row = outs[6].row(i);
        float *kps_row = outs[9].row(i);
        boxes.push_back({box_row[0], box_row[1], box_row[2], box_row[3]});
        kps.push_back({kps_row[0], kps_row[1], kps_row[2], kps_row[3], kps_row[4], kps_row[5], kps_row[6], kps_row[7], kps_row[8], kps_row[9]});
    }
    boxes = box_decode(boxes, grid_40x40, 8);
    kps = kps_decode(kps, grid_40x40, 8);
    for (int i = 0; i < 1600; i++) {
        float score = sqrtf(outs[0].row(i)[0] * outs[3].row(i)[0]);
        if (score > score_threshold) { 
            scores_keep.push_back(score);
            boxes_keep.push_back( boxes[i]);
            kps_keep.push_back(kps[i]);
        } 
    }
    boxes.clear();
    kps.clear();

    // box_stride_16 keypoints_stride_16 meshgrid_20X20
    for (int i = 0; i < 400; i++) {
        float *box_row = outs[7].row(i);
        float *kps_row = outs[10].row(i);
        boxes.push_back({box_row[0], box_row[1], box_row[2], box_row[3]});
        kps.push_back({kps_row[0], kps_row[1], kps_row[2], kps_row[3], kps_row[4], kps_row[5], kps_row[6], kps_row[7], kps_row[8], kps_row[9]});
    }
    boxes = box_decode(boxes, grid_20x20, 16);
    kps = kps_decode(kps, grid_20x20, 16);
    for (int i = 0; i < 400; i++) {
        float score = sqrtf(outs[1].row(i)[0] * outs[4].row(i)[0]);
        if (score > score_threshold) { 
            scores_keep.push_back(score);
            boxes_keep.push_back( boxes[i]);
            kps_keep.push_back(kps[i]);
        }
    }
    boxes.clear();
    kps.clear();

    // box_stride_32 keypoints_stride_32 meshgrid_10X10
    for (int i = 0; i < 100; i++) {
        float *box_row = outs[8].row(i);
        float *kps_row = outs[11].row(i);
        boxes.push_back({box_row[0], box_row[1], box_row[2], box_row[3]});
        kps.push_back({kps_row[0], kps_row[1], kps_row[2], kps_row[3], kps_row[4], kps_row[5], kps_row[6], kps_row[7], kps_row[8], kps_row[9]});
    }
    boxes = box_decode(boxes, grid_10x10, 32);
    kps = kps_decode(kps, grid_10x10, 32);
    for (int i = 0; i < 100; i++) {
        float score = sqrtf(outs[2].row(i)[0] * outs[5].row(i)[0]);
        if (score > score_threshold) { 
            scores_keep.push_back(score);
            boxes_keep.push_back( boxes[i]);
            kps_keep.push_back(kps[i]);
        }
    }
    for (int i = 0; i < boxes_keep.size(); i++) {
        auto &p = boxes_keep[i];
        valid_result.push_back({p.x, p.y, p.w, p.h, scores_keep[i], 0});
    }
    valid_result = nms(valid_result, nms_threshold);
    for (auto &box : valid_result) {
        box.x = (box.x - info.offset_x) / (info.scale + 0.001f);
        box.y = (box.y - info.offset_y) / (info.scale + 0.001f);
        box.w = box.w / (info.scale + 0.001f);
        box.h = box.h / (info.scale + 0.001f);
    }
    return valid_result;
}

