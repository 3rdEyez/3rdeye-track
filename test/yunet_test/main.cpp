#include "yunet_ncnn_detector.h"
#include "opencv2/highgui.hpp"

YunetNCNN detector(
    "E:/Repo/3rdeye-track/models/ncnn/yunet_n_320_320_ncnn_model/yunet_n_320_320.param",
    "E:/Repo/3rdeye-track/models/ncnn/yunet_n_320_320_ncnn_model/yunet_n_320_320.bin",
    true
);

int main(int argc, char const *argv[])
{
    cv::Mat img = cv::imread(TEST_IMAGE_FILE, cv::IMREAD_COLOR);
    auto bboxes = detector.detect(img, 0.4f, 0.5f);
    std::vector<std::string> names{"0"};
    draw_bboxes(img, bboxes, names , 0.8f, 1);
    cv::imshow("yunet", img);
    cv::waitKey(0);
    return 0;
}
