#include "ncnn_detector.h"
#include "detect_utils.h"

bool ncnn_detector::hasGPU = false;

ncnn_detector::ncnn_detector(const char* param, const char* bin, bool useGPU)
{
    this->net = new ncnn::Net();
#if NCNN_USE_GPU
    this->hasGPU = ncnn::get_gpu_count() > 0;
#endif
    this->net->opt.use_vulkan_compute = this->hasGPU && useGPU;
    this->net->load_param(param);
    this->net->load_model(bin);
}

ncnn_detector::~ncnn_detector()
{
    delete this->net;
}

void ncnn_detector::preprocess(const cv::Mat &img, ncnn::Mat &in, letterbox_info &info)
{
    cv::Mat img_copy = img.clone();
    letterbox(img_copy, imgsz[0], imgsz[1], info);
    int img_w = img_copy.cols;
    int img_h = img_copy.rows;
    in = ncnn::Mat::from_pixels(img_copy.data, ncnn::Mat::PIXEL_BGR, img_w, img_h);
    const float mean_vals[3] = { 0.0f, 0.0f, 0.0f };
    const float norm_vals[3] = { 1 / 255.0f, 1 / 255.0f, 1 / 255.0f };
    in.substract_mean_normalize(mean_vals, norm_vals);
}

std::vector<BBox> ncnn_detector::detect(const cv::Mat &img, float score_threshold, float nms_threshold)
{
    ncnn::Mat in, out;
    letterbox_info info;
    this->preprocess(img, in, info);
    auto ex = this->net->create_extractor();
    ex.input("in0", in);
    ex.extract("out0", out);
    std::vector<BBox> result(out.w, { 0, 0, 0, 0, 0, 0 });
    for (int i = 0; i < out.h; i++) {
        float *row = out.row(i);
        for (int j = 0; j < out.w; j++) {
            switch (i) {
                case 0: // x
                    result[j].x = row[j];
                    break;
                case 1: // y
                    result[j].y = row[j];
                    break;
                case 2: // w
                    result[j].w = row[j];
                    break;
                case 3: // h
                    result[j].h = row[j];
                    break;
                default: // cls
                    if (row[j] > result[j].score) {
                        result[j].score = row[j];
                        result[j].cls = i - 4;
                    }
                    break;
            }
        }
    }
    std::vector<BBox> valid_result;
    for (auto &box : result) {
        if (box.score > score_threshold) {
            valid_result.push_back(box);
        }
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

