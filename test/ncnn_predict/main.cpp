#include <net.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "image_utils.h"
#include "ncnn_detector.h"

int main(int argc, char const *argv[])
{
    cv::Mat img = cv::imread("F:/bus.jpg");
    cv::imshow("origin", img);
    int off_x, off_y;
    float scale;
    letterbox(img, 640, 640, off_x, off_y, scale);
    cv::imshow("letterbox", img);
    cv::imwrite("F:/bus_letterbox.jpg", img);
    printf("scale: %f, off_x: %d, off_y: %d\n", scale, off_x, off_y);

    ncnn_detector detector(
        "E:/Repo/3rdeye-track/model/ncnn/yolov8m.param",
        "E:/Repo/3rdeye-track/model/ncnn/yolov8m.bin",
        true
    );
    auto bboxes = detector.detect(img, 0.2f, 0.5f);
    for (auto &bbox : bboxes) {
        printf("%f %f %f %f %f %d\n", bbox.x, bbox.y, bbox.w, bbox.h, bbox.score, bbox.cls);
        int x1 = (int)(bbox.x - bbox.w / 2);
        int y1 = (int)(bbox.y - bbox.h / 2);
        int x2 = (int)(bbox.x + bbox.w / 2);
        int y2 = (int)(bbox.y + bbox.h / 2);

        cv::rectangle(img, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 0, 255), 2);
    }
    cv::imshow("result", img);
    cv::waitKey(0);
    return 0;
}
