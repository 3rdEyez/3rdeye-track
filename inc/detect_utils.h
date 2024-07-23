#ifndef __DETECT_UTILS_H__
#define __DETECT_UTILS_H__

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>
#include <queue>
#include <unordered_map>
#include <cstdint>
#include <iomanip>
#include <sstream>
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


typedef std::deque<BBox> track_t;
typedef std::unordered_map<uint64_t, track_t> tracks_t;

const int color_list[80][3] = {
    //{255, 255, 255}, //bg
    {216,  82,  24}, {236, 176,  31}, {125,  46, 141}, {118, 171,  47}, { 76, 189, 237},
    {238,  19,  46}, { 76,  76,  76}, {153, 153, 153}, {255,   0,   0}, {255, 127,   0},
    {190, 190,   0}, {  0, 255,   0}, {  0,   0, 255}, {170,   0, 255}, { 84,  84,   0},
    { 84, 170,   0}, { 84, 255,   0}, {170,  84,   0}, {170, 170,   0}, {170, 255,   0},
    {255,  84,   0}, {255, 170,   0}, {255, 255,   0}, {  0,  84, 127}, {  0, 170, 127},
    {  0, 255, 127}, { 84,   0, 127}, { 84,  84, 127}, { 84, 170, 127}, { 84, 255, 127},
    {170,   0, 127}, {170,  84, 127}, {170, 170, 127}, {170, 255, 127}, {255,   0, 127},
    {255,  84, 127}, {255, 170, 127}, {255, 255, 127}, {  0,  84, 255}, {  0, 170, 255},
    {  0, 255, 255}, { 84,   0, 255}, { 84,  84, 255}, { 84, 170, 255}, { 84, 255, 255},
    {170,   0, 255}, {170,  84, 255}, {170, 170, 255}, {170, 255, 255}, {255,   0, 255},
    {255,  84, 255}, {255, 170, 255}, { 42,   0,   0}, { 84,   0,   0}, {127,   0,   0},
    {170,   0,   0}, {212,   0,   0}, {255,   0,   0}, {  0,  42,   0}, {  0,  84,   0},
    {  0, 127,   0}, {  0, 170,   0}, {  0, 212,   0}, {  0, 255,   0}, {  0,   0,  42},
    {  0,   0,  84}, {  0,   0, 127}, {  0,   0, 170}, {  0,   0, 212}, {  0,   0, 255},
    {  0,   0,   0}, { 36,  36,  36}, { 72,  72,  72}, {109, 109, 109}, {145, 145, 145},
    {182, 182, 182}, {218, 218, 218}, {  0, 113, 188}, { 80, 182, 188}, {127, 127,   0},
};


void letterbox(cv::Mat& img, int new_width, int new_height, letterbox_info& info);
std::vector<BBox> nms(std::vector<BBox> &bboxes, float iou_threshold);
std::vector<BBox> bbox_select(const std::vector<BBox> &bboxes, uint32_t cls_to_select);
BBox_xyxy xywh2xyxy(BBox_xywh box);
float bbox_iou(BBox_xyxy box1, BBox_xyxy box2);
BBox xywh2xyxy(float x, float y, float w, float h);
void draw_box_in_color(cv::Mat& img, Box box, cv::Scalar color, int thickness);
void draw_bboxes(cv::Mat& img, const std::vector<BBox> &bboxes, std::vector<std::string> &classes, float text_scale, int thickness);
void draw_tracks(cv::Mat& img, tracks_t &tracks, int thickness);
#endif //__DETECT_UTILS_H__
