#ifndef __3RDEYE_COMMON_H__

#include <vector>
#include <cstdint>


typedef struct {
    float x;
    float y;
    float w;
    float h;
    float score;
    uint32_t cls;
}BBox;

typedef struct {
    uint32_t x;
    uint32_t y;
    uint32_t w;
    uint32_t h;
}xywh_box;


typedef struct {
    uint32_t x1;
    uint32_t y1;
    uint32_t x2;
    uint32_t y2;
} xyxy_box;

#define __3RDEYE_COMMON_H__
#endif /* __3RDEYE_COMMON_H__ */
