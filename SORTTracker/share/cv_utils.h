#ifndef __3RDEYE_TRACK_CV_UTILS_H__
#define __3RDEYE_TRACK_CV_UTILS_H__

#include "dtype.h"
#include <opencv2/core.hpp>

void letterbox(cv::Mat& img, int new_width, int new_height, letterbox_info& info);
void draw_box_in_color(cv::Mat& img, Box box, cv::Scalar color, int thickness);
void draw_bboxes(cv::Mat& img, const std::vector<BBox> &bboxes, std::vector<std::string> &classes, float text_scale, int thickness);
void draw_tracks(cv::Mat& img, const trackmap_t &tracks, int thickness);

#endif // __3RDEYE_TRACK_CV_UTILS_H__
