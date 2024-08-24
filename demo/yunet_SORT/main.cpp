#include <iostream>
#include <chrono>
#include "yunet_ncnn_detector.h"
#include "opencv2/highgui.hpp"
#include "tracker.h"

int main(int argc, char const *argv[])
{
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <ncnn.param> <ncnn.bin> <video_path>" << std::endl;
        return -1;
    }
    YunetNCNN detector(argv[1], argv[2], true);
    cv::Mat frame;
    cv::VideoCapture cap;
    if (strcmp(argv[3], "0") == 0)
        cap = cv::VideoCapture(0);
    else
        cap = cv::VideoCapture(argv[3]);
    double fps = cap.get(cv::CAP_PROP_FPS);
#if SAVE_VIDEO
    cap >> frame; // 读取第一帧git
    cv::VideoWriter writer("out.mp4", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), fps, frame.size());
#endif
    int tracker = tracker_init(2.8f, 2.0f, 20, 8);
    auto start = std::chrono::steady_clock::now();
    auto end = std::chrono::steady_clock::now();

    if (!cap.isOpened()) return -1;
    while (true)
    {   
        start = std::chrono::steady_clock::now();
        cap.read(frame); // 读取一帧
        if(frame.empty()) {
            std::cerr << "Error: Blank frame grabbed" << std::endl;
            break;
        }
        auto bboxes = detector.detect(frame, 0.7f, 0.5f);
        tracker_update(tracker, bboxes);
        auto all_tracks = tracker_get_tracks(tracker);
        draw_tracks(frame, all_tracks, 1);
#if SAVE_VIDEO
        writer.write(frame);
#endif
#if IMSHOW
        cv::imshow("frame", frame);
        if (cv::waitKey(1) == 27) break;
#endif
        end = std::chrono::steady_clock::now();
        printf("fps = %f\n", 1.0 / std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count());
    }
    cap.release();

#if SAVE_VIDEO
    writer.release();
#endif
    return 0;
}
