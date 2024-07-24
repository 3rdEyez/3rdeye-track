#include <iostream>
#include <unordered_map>
#include <queue>
#include <set>
#include <cmath>
#include <opencv2/highgui/highgui.hpp>
#include "ncnn_detector.h"
#include "Hungarian.h"
#include "kalman_filter.h"
#include "sort_tracker.h"

constexpr int MAX_AGE=20;
constexpr int TRACK_MAX_LEN=20;
constexpr float SIGMA_Q = 4.1f;
constexpr float SIGMA_R = 5.0f;
constexpr float DETECT_CONF_THRESH = 0.4f;
constexpr float DETECT_NMS_IOU_THRESH = 0.5f;

#ifdef SAVE_VIDEO
std::string outputVideoPath = "output_video.mp4";
int codec = cv::VideoWriter::fourcc('a', 'v', 'c', '1'); // 编码器
#endif

int main(int argc, char const *argv[])
{
    cv::Mat frame;
    cv::VideoCapture cap(TEST_VIDEO_FILE);
    std::unordered_map<uint64_t, KalmanFilter*> kf_map; // 存储每个track的KalmanFilter
    std::unordered_map<uint64_t, uint64_t> track_unmatched_cnt_map; // 存储每个track的未匹配帧数
    std::vector<BBox> new_track_bboxes; // 存储bbox, 用于创建新轨迹
    uint64_t id_num = 0;
    uint64_t frame_id = 0;

    ncnn_detector detector(
        YOLOV8_NCNN_MODEL"/model.ncnn.param",
        YOLOV8_NCNN_MODEL"/model.ncnn.bin",
        true
    );

    sort_tracker tracker(
        SIGMA_Q,
        SIGMA_R,
        TRACK_MAX_LEN,
        MAX_AGE
    );

    if(!cap.isOpened()) {
        std::cerr << "Error: Could not open camera" << std::endl;
        return -1;
    }

#ifdef SAVE_VIDEO
    cv::Size frameSize = cv::Size((int)cap.get(cv::CAP_PROP_FRAME_WIDTH), (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    double fps = cap.get(cv::CAP_PROP_FPS);
    cv::VideoWriter videoWriter(outputVideoPath, codec, fps, frameSize);
#endif
    while(true) {
        cap.read(frame); // 读取一帧
        if(frame.empty()) {
            std::cerr << "Error: Blank frame grabbed" << std::endl;
            break;
        }
        // 在frame上显示帧号
        cv::putText(frame, "frame: " + std::to_string(++frame_id), cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 2);
        // 检测
        cv::Mat frame_rgb;
        cv::cvtColor(frame, frame_rgb, cv::COLOR_BGR2RGB);
        auto BBoxes = detector.detect(frame_rgb, DETECT_CONF_THRESH, DETECT_NMS_IOU_THRESH);
        BBoxes = bbox_select(BBoxes, 0); // 只挑出类别为0的目标(Persion)
        tracker.exec(BBoxes);
        auto all_tracks = tracker.get_all_tracks();

        // 绘制检测框
        // draw_bboxes(frame, BBoxes, detector.names, 0.5f, 1);
        
        // 绘制轨迹
        draw_tracks(frame, all_tracks, 2);
        // 显示帧
        cv::imshow("Frame", frame);
#ifdef SAVE_VIDEO
        videoWriter.write(frame);
#endif
        // 按 'ESC' 键退出循环
        if(cv::waitKey(1) == 27) {
            break;
        }
    }
    cap.release();
    cv::destroyAllWindows();
#ifdef SAVE_VIDEO
    videoWriter.release();
#endif
    return 0;
}
