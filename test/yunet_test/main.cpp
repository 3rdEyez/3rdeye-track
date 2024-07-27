#include <iostream>
#include "yunet_ncnn_detector.h"
#include "opencv2/highgui.hpp"
#include "sort_tracker.h"

YunetNCNN detector(
    "E:/Repo/3rdeye-track/models/ncnn/yunet_n_320_320_ncnn_model/yunet_n_320_320.param",
    "E:/Repo/3rdeye-track/models/ncnn/yunet_n_320_320_ncnn_model/yunet_n_320_320.bin",
    true
);

int main(int argc, char const *argv[])
{
    cv::Mat frame;
    cv::VideoCapture cap(0);
    sort_tracker tracker(0.1f, 1.0f, 20, 20);

    if (!cap.isOpened()) return -1;
    while (true)
    {    
        cap.read(frame); // 读取一帧
        if(frame.empty()) {
            std::cerr << "Error: Blank frame grabbed" << std::endl;
            break;
        }
        auto bboxes = detector.detect(frame, 0.6f, 0.5f);
        tracker.exec(bboxes);
        auto all_tracks = tracker.get_all_tracks();
        draw_tracks(frame, all_tracks, 1);
        cv::imshow("yunet", frame);
        if (cv::waitKey(1) == 27) break; // 按下ESC键退出
    }
    cap.release();
    cv::destroyAllWindows();
    return 0;
}
