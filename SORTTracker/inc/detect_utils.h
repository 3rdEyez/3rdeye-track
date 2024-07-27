#ifndef __DETECT_UTILS_H__
#define __DETECT_UTILS_H__

#include <opencv2/core.hpp>
#include <vector>
#include <queue>
#include <unordered_map>
#include <cstdint>
#include <string>


typedef struct {
    float x;
    float y;
    float w;
    float h;
}Box;

typedef struct {
    float x;
    float y;
    float w;
    float h;
    float score;
    uint32_t cls;
}BBox;


// default BBox type is xywh
typedef BBox BBox_xywh;


typedef struct {
    float x1;
    float y1;
    float x2;
    float y2;
    float score;
    uint32_t cls;
}BBox_xyxy;


typedef struct {
    int offset_x;
    int offset_y;
    float scale;
} letterbox_info;



void letterbox(cv::Mat& img, int new_width, int new_height, letterbox_info& info);
std::vector<BBox> nms(std::vector<BBox> &bboxes, float iou_threshold);
std::vector<BBox> bbox_select(const std::vector<BBox> &bboxes, uint32_t cls_to_select);
BBox_xyxy xywh2xyxy(BBox_xywh box);
float bbox_iou(BBox_xyxy box1, BBox_xyxy box2);
BBox xywh2xyxy(float x, float y, float w, float h);
void draw_box_in_color(cv::Mat& img, Box box, cv::Scalar color, int thickness);
void draw_bboxes(cv::Mat& img, const std::vector<BBox> &bboxes, std::vector<std::string> &classes, float text_scale, int thickness);

#endif //__DETECT_UTILS_H__
