#include <iostream>
#include <chrono>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "ncnn_detector.h"
#include "tracker.h"
#include "utils/cv_utils.h"


int main(int argc, char const *argv[])
{
    
    char frame_path_fmt[] = "testvideo_frames/%05d.jpg";
    char out_frame_path_fmt[] = "out_frames/%05d.jpg";
    ncnn_detector detector(
        "yolov8n_ncnn_model_fp16/model.ncnn.param",
        "yolov8n_ncnn_model_fp16/model.ncnn.bin",
        true
    );
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
