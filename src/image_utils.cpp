#include "image_utils.h"


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