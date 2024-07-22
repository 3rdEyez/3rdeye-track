#include <opencv2/imgproc.hpp>
#include "detect_utils.h"

void letterbox(cv::Mat& img, int new_width, int new_height, int &offset_x, int &offset_y, float &scale)
{
    int width = img.cols;
    int height = img.rows;
    scale = std::min((float)new_width / (float)width, (float)new_height / (float)height);
    int new_unscaled_width = (int)(scale * (float)width);
    int new_unscaled_height = (int)(scale * (float)height);
    offset_x = (new_width - new_unscaled_width) / 2;
    offset_y = (new_height - new_unscaled_height) / 2;
    cv::resize(img, img, cv::Size(new_unscaled_width, new_unscaled_height));
    cv::copyMakeBorder(img, img, offset_y, offset_y, offset_x, offset_x, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
}


BBox_xyxy xywh2xyxy(BBox_xywh box)
{
    float x = box.x;
    float y = box.y;
    float w = box.w;
    float h = box.h;
    return BBox_xyxy{x - w / 2, y - h / 2, x + w / 2, y + h / 2};
}


float bbox_iou(BBox_xyxy box1, BBox_xyxy box2)
{
    float ax1 = box1.x1, ay1 = box1.y1, ax2 = box1.x2, ay2 = box1.y2;
    float bx1 = box2.x1, by1 = box2.y1, bx2 = box2.x2, by2 = box2.y2;
    float sa = (ax2 - ax1) * (ay2 - ay1);
    float sb = (bx2 - bx1) * (by2 - by1);
    if (ax1 >= bx2 || ax2 <= bx1 || ay1 >= by2 || ay2 <= by1)
        return 0.0f;
    float x_list[4] = {ax1, bx1, ax2, bx2};
    float y_list[4] = {ay1, by1, ay2, by2};
    std::sort(x_list, x_list + 4);
    std::sort(y_list, y_list + 4);
    float inter_area = (x_list[2] - x_list[1]) * (y_list[2] - y_list[1]);
    return inter_area / (sa + sb - inter_area);
}

std::vector<BBox> nms(std::vector<BBox> &bboxes, float iou_threshold)
{   
    std::vector<BBox> result;
    std::unordered_map<int, std::vector<BBox>> in_map, out_map; // class label as key, bbox list as value
    for (auto &bbox : bboxes) in_map[bbox.cls].push_back(bbox);

    for (auto &item : in_map) {
        auto &bboxes = item.second;
        auto cls = item.first;
        std::sort(bboxes.begin(), bboxes.end(), [](const BBox &a, const BBox &b) { return a.score > b.score; });
        for (int i = 0; i < bboxes.size(); i++) {
            bool is_valid = true;
            auto &saved_bboxes = out_map[cls];
            for (int j = 0; j < saved_bboxes.size(); j++) {
                float iou = bbox_iou(xywh2xyxy(bboxes[i]), xywh2xyxy(saved_bboxes[j]));
                if (iou > iou_threshold) {
                    is_valid = false;
                    break;
                }
            }
            if (is_valid) saved_bboxes.push_back(bboxes[i]);
        }
        // Complete the Non-Maximum Suppression (NMS) for a category.
        for (auto &bbox : out_map[cls]) result.push_back(bbox);
    }
    
    return result;
}