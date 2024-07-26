
#ifndef __YUNET_NCNN_DETECTOR_H__
#define __YUNET_NCNN_DETECTOR_H__

#include <vector>
#include "net.h"
#include "detect_utils.h"
#include "opencv2/core.hpp"

class YunetNCNN
{
public:
    YunetNCNN(const char* param, const char* bin, bool useGPU);
    ~YunetNCNN();

    ncnn::Net *net;
    bool hasGPU;
    int imgsz[2] = {320, 320};
    std::vector<BBox> detect(const cv::Mat &img, float score_threshold, float nms_threshold);
private:
    void preprocess(const cv::Mat &img, ncnn::Mat &in, letterbox_info &info);
};


typedef std::vector<float> Vec1f;
typedef std::vector<Vec1f> Vec2f;
typedef std::vector<Vec2f> Vec3f;

#endif // __YUNET_NCNN_DETECTOR_H__
