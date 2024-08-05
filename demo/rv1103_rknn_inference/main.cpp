#include "yunet_rknn_detector.h"
#include <opencv2/highgui.hpp>
#include "cv_utils.h"
#include <iostream>
#include <chrono>

constexpr int LOOP_N = 100;

int main(int argc, char const *argv[])
{
    cv::VideoCapture cap;
    cap.open(0);
    cv::Mat img;
    YunetRKNN detector("yunet_n_320_320.rknn");
    cap >> img;
    if (img.empty()) {
        std::cerr << "open camera failed" << std::endl;
        return -1;
    }
    auto bboxes = detector.detect(img, 0.8, 0.4);
    auto start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < LOOP_N; i++) {
        cap >> img;
        bboxes = detector.detect(img, 0.8, 0.4);
        draw_bboxes(img, bboxes, detector.names, 0.8, 1);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "average time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / LOOP_N << "ms" << std::endl;
    cv::imwrite("result.jpg", img);
    return 0;
}
