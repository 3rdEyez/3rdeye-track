#include "ncnn_detector.h"
#include "detect_utils.h"

bool ncnn_detector::hasGPU = false;

ncnn_detector::ncnn_detector(const char* param, const char* bin, bool useGPU)
{
    this->net = new ncnn::Net();
#ifdef NCNN_USE_GPU
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

void ncnn_detector::preprocess(const cv::Mat &img, ncnn::Mat &in)
{
    int img_w = img.cols;
    int img_h = img.rows;

    in = ncnn::Mat::from_pixels(img.data, ncnn::Mat::PIXEL_BGR, img_w, img_h);
    // const float mean_vals[3] = { 103.53f, 116.28f, 123.675f };
    // const float norm_vals[3] = { 0.017429f, 0.017507f, 0.017125f };
    const float mean_vals[3] = { 128.0f, 128.0f, 128.0f };
    const float norm_vals[3] = { 1 / 128.0f, 1 / 128.0f, 1 / 128.0f };
    in.substract_mean_normalize(mean_vals, norm_vals);
}

std::vector<BBox> ncnn_detector::detect(const cv::Mat &img, float score_threshold, float nms_threshold)
{
    ncnn::Mat in, out;
    this->preprocess(img, in);
    auto ex = this->net->create_extractor();
#ifdef NCNN_USE_GPU
    ex.set_vulkan_compute(this->hasGPU);
#endif
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
    return nms(valid_result, nms_threshold);
}

