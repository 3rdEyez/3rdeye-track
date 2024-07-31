#include <iostream>
#include <chrono>
#include <thread>
#include <mutex>
#include <queue>
#include "yunet_ncnn_detector.h"
#include "opencv2/highgui.hpp"
#include "tracker.h"

std::mutex mtx;
std::queue<cv::Mat> frame_queue;

int main(int argc, char const *argv[])
{
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <ncnn.param> <ncnn.bin> <video_path>" << std::endl;
        return -1;
    }
    YunetNCNN detector(argv[1], argv[2], true);
    cv::Mat frame;
    cv::VideoCapture cap(argv[3]);
    cap >> frame; // 读取第一帧
    cv::VideoWriter writer("out.avi", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 30, frame.size());
    int tracker = tracker_init(0.1f, 1.0f, 20, 20);
    auto start = std::chrono::steady_clock::now();
    auto end = std::chrono::steady_clock::now();

    std::thread writer_thread([&]() {
        static int cnt = 0;
        while (1) {
            mtx.lock();
            if(frame_queue.empty()) {
                if (cnt > 100) {
                    mtx.unlock();
                    break;
                }
                mtx.unlock();
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                cnt += 1;
                continue;
            }
            cv::Mat frame = frame_queue.front();
            if (frame.empty()) {
                mtx.unlock();
                break;
            }
            frame_queue.pop();
            std::cout << " length = " << frame_queue.size() << std::endl;
            mtx.unlock();
            writer.write(frame);
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
            cnt = 0;
        }
    });

    if (!cap.isOpened()) return -1;
    while (true)
    {   
        start = std::chrono::steady_clock::now();
        cap.read(frame); // 读取一帧
        if(frame.empty()) {
            std::cerr << "Error: Blank frame grabbed" << std::endl;
            break;
        }
        auto bboxes = detector.detect(frame, 0.6f, 0.5f);
        tracker_update(tracker, bboxes);
        auto all_tracks = tracker_get_tracks(tracker);
        draw_tracks(frame, all_tracks, 1);
        mtx.lock();
        frame_queue.push(frame);
        mtx.unlock();
        end = std::chrono::steady_clock::now();
        printf("fps = %f\n", 1.0 / std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count());
    }
    cap.release();
    // waitting for video_writer thread to finish
    writer_thread.join();
    writer.release();
    return 0;
}

/*
    char frame_path_fmt[] = "testvideo_frames/%05d.jpg";
    char out_frame_path_fmt[] = "out_frames/%05d.jpg";
    cv::Mat frame;
    trackmap_t all_tracks;
    int trackerid = tracker_init(0.8f, 2.0f, 20, 20);
    
    int i = 0;
    while (true)
    {
        auto start = std::chrono::steady_clock::now();
        std::vector<BBox> bboxes;
        frame = cv::imread(cv::format(frame_path_fmt, ++i));
        if (frame.empty())
            break;
        bboxes = detector.detect(frame, 0.2, 0.5);
        auto end = std::chrono::steady_clock::now();
        printf("frame %d, %d bboxes ; fps = %f\n", i, (int)bboxes.size(), 1.0 / std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count());
        tracker_update(trackerid, bbox_select(bboxes, 2));
        all_tracks = tracker_get_tracks(trackerid);
        draw_tracks(frame, all_tracks, 1);
        cv::imwrite(cv::format(out_frame_path_fmt, i), frame);
    }

    return 0;
*/