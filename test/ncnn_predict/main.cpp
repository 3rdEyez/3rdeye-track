#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "utils/utils.h"
#include "ncnn_detector.h"

int main(int argc, char const *argv[])
{
    cv::Mat img = cv::imread(TEST_IMAGE_FILE);
    cv::imshow("origin", img);

    ncnn_detector detector(
        YOLOV8_NCNN_MODEL".param",
        YOLOV8_NCNN_MODEL".bin",
        true
    );
    auto bboxes = detector.detect(img, 0.2f, 0.4f);
    for (auto &bbox : bboxes) {
        printf("%f %f %f %f %f %d\n", bbox.x, bbox.y, bbox.w, bbox.h, bbox.score, bbox.cls);
        int x1 = (int)(bbox.x - bbox.w / 2);
        int y1 = (int)(bbox.y - bbox.h / 2);
        int x2 = (int)(bbox.x + bbox.w / 2);
        int y2 = (int)(bbox.y + bbox.h / 2);
        auto color = color_list[bbox.cls % 80];
        cv::rectangle(img, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(color[0], color[1], color[2], 1), 2);
    }
    cv::imshow("result", img);
    cv::waitKey(0);
    return 0;
}
