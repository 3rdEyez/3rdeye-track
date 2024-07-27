#include <iostream>
#include <chrono>
#include "yunet_ncnn_detector.h"
#include "opencv2/highgui.hpp"
#include "tracker.h"

YunetNCNN detector(
    "E:/Repo/3rdeye-track/models/ncnn/yunet_n_320_320_ncnn_model/yunet_n_320_320.param",
    "E:/Repo/3rdeye-track/models/ncnn/yunet_n_320_320_ncnn_model/yunet_n_320_320.bin",
    true
);

int main(int argc, char const *argv[])
{
    // cv::Mat frame;
    // cv::VideoCapture cap(0);
    // sort_tracker tracker(0.1f, 1.0f, 20, 20);

    // if (!cap.isOpened()) return -1;
    // while (true)
    // {    
    //     cap.read(frame); // 读取一帧
    //     if(frame.empty()) {
    //         std::cerr << "Error: Blank frame grabbed" << std::endl;
    //         break;
    //     }
    //     auto bboxes = detector.detect(frame, 0.6f, 0.5f);
    //     tracker.exec(bboxes);
    //     auto all_tracks = tracker.get_all_tracks();
    //     draw_tracks(frame, all_tracks, 1);
    //     cv::imshow("yunet", frame);
    //     if (cv::waitKey(1) == 27) break; // 按下ESC键退出
    // }
    // cap.release();
    // cv::destroyAllWindows();
    // return 0;

    
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
}
