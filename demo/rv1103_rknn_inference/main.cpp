#include "yunet_rknn_detector.h"
#include <opencv2/highgui.hpp>
#include "cv_utils.h"
#include <iostream>
#include <chrono>
#include <thread>
#include <mutex>
#include <queue>

std::mutex mtx;
typedef std::pair<int, cv::Mat> payload_t;

constexpr int LOOP_N = 500;

int main(int argc, char const *argv[])
{
    cv::VideoCapture cap;
    cap.open(0);
    cv::Mat img;
    YunetRKNN detector("yunet_n_320_320.rknn");
    std::deque<payload_t> imgQ;
    cap >> img;
    if (img.empty()) {
        std::cerr << "open camera failed" << std::endl;
        return -1;
    }

    std::thread producer([&]() {
        for (int i = 0; i < LOOP_N; i++) {
            mtx.lock();
            if (imgQ.size() > 10) {
                mtx.unlock();
                continue;
            }
            mtx.unlock();
            cap >> img;
            mtx.lock();
            imgQ.push_back(std::make_pair(i, img));
            mtx.unlock();
        }
    });

    auto start = std::chrono::high_resolution_clock::now();
    while (true) {
        cv::Mat img;
        mtx.lock();
        if (imgQ.empty()) {
            mtx.unlock();
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }
        auto p = imgQ.front();
        imgQ.pop_front();
        mtx.unlock();
        img = p.second;
        if (img.empty()) {
            std::cerr << "read frame failed" << std::endl;
            return -1;
        }
        auto bboxes = detector.detect(img, 0.8, 0.5);
        draw_bboxes(img, bboxes, detector.names, 0.8, 1);
        if (p.first == LOOP_N - 1) {
            cv::imwrite("result.jpg", img);
            std::cout << "loop end" << std::endl;
            break;
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "average time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / LOOP_N << "ms" << std::endl;
    producer.join();
    return 0;
}
