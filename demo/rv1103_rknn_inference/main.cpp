#include "yunet_rknn_detector.h"
#include <opencv2/highgui.hpp>
#include "cv_utils.h"
#include <iostream>
#include <chrono>

constexpr int LOOP_N = 1000;

int main(int argc, char const *argv[])
{
    YunetRKNN detector("yunet_n_320_320.rknn");
    cv::Mat img = cv::imread("bus.jpg", cv::IMREAD_COLOR);
    auto bboxes = detector.detect(img, 0.8, 0.4);
    auto start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < LOOP_N; i++) {
        bboxes = detector.detect(img, 0.8, 0.4);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "average time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / LOOP_N << "ms" << std::endl;
    
    draw_bboxes(img, bboxes, detector.names, 0.8, 1);
    cv::imwrite("result.jpg", img);
    return 0;
}
