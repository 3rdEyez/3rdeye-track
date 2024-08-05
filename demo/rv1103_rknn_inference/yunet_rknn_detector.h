#ifndef __YUNET_RKNN_DETECTOR_H__
#define __YUNET_RKNN_DETECTOR_H__

#include "rknn_api.h"
#include "dtype.h"
#include "opencv2/core.hpp"
#include <vector>

#ifdef RV1106_1103
    typedef struct {
        char *dma_buf_virt_addr;
        int dma_buf_fd;
        int size;
    }rknn_dma_buf;
#endif 

typedef struct {
    rknn_context rknn_ctx;
    rknn_input_output_num io_num;
    rknn_tensor_attr* input_attrs;
    rknn_tensor_attr* output_attrs;
#if defined(RV1106_1103) 
    rknn_tensor_mem* input_mems[1];
    rknn_tensor_mem* output_mems[9];
    rknn_dma_buf img_dma_buf;
#endif
    int model_channel;
    int model_width;
    int model_height;
    bool is_quant;
} rknn_app_context_t;


class YunetRKNN
{
public:
    YunetRKNN(const char *rknn_model);
    ~YunetRKNN();
    rknn_app_context_t app_ctx;
    int imgsz[2] = {320, 320};
    std::vector<BBox> detect(const cv::Mat &img, float score_threshold, float nms_threshold);
    std::vector<std::string> names{"person"};
private:
    void preprocess(const cv::Mat &img, cv::Mat &in, letterbox_info &info);
};

typedef std::vector<float> Vec1f;
typedef std::vector<Vec1f> Vec2f;
typedef std::vector<Vec2f> Vec3f;

#endif // __YUNET_RKNN_DETECTOR_H__