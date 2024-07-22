#ifndef __3RDEYE_COMMON_H__

#include <vector>
#include <unordered_map>
#include <cstdint>


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



#define __3RDEYE_COMMON_H__
#endif /* __3RDEYE_COMMON_H__ */
