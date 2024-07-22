#ifndef __DETECT_UTILS_H__
#define __DETECT_UTILS_H__

#include "common.h"
#include <opencv2/highgui.hpp>

void letterbox(cv::Mat& img, int new_width, int new_height, int &offset_x, int &offset_y, float &scale);
std::vector<BBox> nms(std::vector<BBox> &bboxes, float iou_threshold);

#endif //__DETECT_UTILS_H__
