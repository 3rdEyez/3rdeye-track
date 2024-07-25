#ifndef __NCNN_DETECTOR_H__
#define __NCNN_DETECTOR_H__

#include <net.h>
#include <opencv2/core/core.hpp>
#include "detect_utils.h"

class ncnn_detector
{
public:
    ncnn_detector(const char* param, const char* bin, bool useGPU);
    ~ncnn_detector();

    ncnn::Net *net;
    static bool hasGPU;
    int imgsz[2] = {640, 640};
    int num_class = 80;
    std::vector<BBox> detect(const cv::Mat &img, float score_threshold, float nms_threshold);
    std::vector<std::string> names { 
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
        "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
        "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table+", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
        "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
        "hair drier", "toothbrush" };

private:
    void preprocess(const cv::Mat &img, ncnn::Mat &in, letterbox_info &info);
};

#endif // __NCNN_DETECTOR_H__

