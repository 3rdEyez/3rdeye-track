#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "det_utils.h"
#include "ncnn_detector.h"

int main(int argc, char const *argv[])
{
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <ncnn.param> <ncnn.bin> <image_path>" << std::endl;
        return -1;
    }
    
    ncnn_detector detector(argv[1], argv[2], true);

    std::cout << "image file:" << argv[3] << std::endl;
    cv::Mat img = cv::imread(argv[3], cv::IMREAD_COLOR);
    cv::imshow("origin", img);

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
