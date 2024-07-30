#ifndef __SORTTRACKER_DTYPE_H__
#define __SORTTRACKER_DTYPE_H__

#include <deque>
#include <unordered_map>
#include <cstdint>

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
    int cls;
}BBox;

// default BBox type is xywh
typedef BBox BBox_xywh;

typedef struct {
    float x1;
    float y1;
    float x2;
    float y2;
    float score;
    int cls;
}BBox_xyxy;


typedef struct {
    int offset_x;
    int offset_y;
    float scale;
} letterbox_info;

typedef std::deque<BBox> track_t;
typedef std::unordered_map<uint64_t, track_t> trackmap_t;

#endif // __SORTTRACKER_DTYPE_H__
