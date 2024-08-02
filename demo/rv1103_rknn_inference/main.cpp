#include "yunet_rknn_detector.h"
#include <opencv2/highgui.hpp>

int main(int argc, char const *argv[])
{
    YunetRKNN detector("yunet_n_320_320.rknn");
    cv::Mat img = cv::imread("bus.jpg", cv::IMREAD_COLOR);
    detector.detect(img, 0.4, 0.4);

    return 0;
}
