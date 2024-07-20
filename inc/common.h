#ifndef __3RDEYE_COMMON_H__

#include <vector>
#include <cstdint>

/**
 * @brief bounding box structure
 * x: x coordinate of centor of the bounding box
 * y: y coordinate of centor of the bounding box
 * w: width of the bounding box
 * h: height of the bounding box
 * score: confidence score of the bounding box
 * cls: class label of the bounding box
 */
typedef struct {
    float x;
    float y;
    float w;
    float h;
    float score;
    uint32_t cls;
}BBox;


/**
 * @brief bounding box structure
 * x: x coordinate of the centor of the bounding box
 * y: y coordinate of the centor of the bounding box
 * w: width of the bounding box
 * h: height of the bounding box
 */
typedef struct {
    uint32_t x;
    uint32_t y;
    uint32_t w;
    uint32_t h;
}xywh_box;


/**
 * @brief bounding box structure
 * x1: x coordinate of the top left corner of the bounding box
 * y1: y coordinate of the top left corner of the bounding box
 * x2: x coordinate of the bottom right corner of the bounding box
 * y2: y coordinate of the bottom right corner of the bounding box
 */
typedef struct {
    uint32_t x1;
    uint32_t y1;
    uint32_t x2;
    uint32_t y2;
} xyxy_box;

#define __3RDEYE_COMMON_H__
#endif /* __3RDEYE_COMMON_H__ */
