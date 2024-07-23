#include <iostream>
#include <unordered_map>
#include <queue>
#include <set>
#include <cmath>
#include <opencv2/highgui/highgui.hpp>
#include "ncnn_detector.h"
#include "Hungarian.h"
#include "kalman_filter.h"

constexpr int MAX_AGE=20;
constexpr int TRACK_MAX_LEN=20;
constexpr float SIGMA_Q = 0.01f;
constexpr float SIGMA_R = 2.0f;
vector<uint64_t> assignment_debug;

int main(int argc, char const *argv[])
{
    cv::Mat frame;
    cv::VideoCapture cap(TEST_VIDEO_FILE);
    tracks_t all_tracks;
    std::unordered_map<uint64_t, KalmanFilter*> kf_map; // 存储每个track的KalmanFilter
    std::unordered_map<uint64_t, uint64_t> track_unmatched_cnt_map; // 存储每个track的未匹配帧数
    uint64_t id_num = 0;

    ncnn_detector detector(
        YOLOV8_NCNN_MODEL".param",
        YOLOV8_NCNN_MODEL".bin",
        true
    );

    if(!cap.isOpened()) {
        std::cerr << "Error: Could not open camera" << std::endl;
        return -1;
    }

    while(true) {
        cap.read(frame); // 读取一帧

        if(frame.empty()) {
            std::cerr << "Error: Blank frame grabbed" << std::endl;
            break;
        }
        // 检测
        cv::Mat frame_rgb;
        cv::cvtColor(frame, frame_rgb, cv::COLOR_BGR2RGB);
        auto BBoxes = detector.detect(frame_rgb, 0.4f, 0.2f);
        BBoxes = bbox_select(BBoxes, 0); // 只挑出类别为0的目标(Persion)
        // 如果tracks为空 则初始化
        if(all_tracks.empty()) {
            for(auto &bbox : BBoxes) {
                track_t track;
                track.push_back(bbox);
                all_tracks[id_num] = track;

                auto state = Eigen::VectorXf(7);
                float x = bbox.x, y = bbox.y, s = bbox.w * bbox.h, r = bbox.w / (bbox.h + 1e-5f);
                state << x, y, s, r, 0, 0, 0;
                auto A = Eigen::MatrixXf(7, 7);
                A << 1, 0, 0, 0, 1, 0, 0,
                     0, 1, 0, 0, 0, 1, 0,
                     0, 0, 1, 0, 0, 0, 1,
                     0, 0, 0, 1, 0, 0, 0,
                     0, 0, 0, 0, 1, 0, 0,
                     0, 0, 0, 0, 0, 1, 0,
                     0, 0, 0, 0, 0, 0, 1;
                auto B = Eigen::MatrixXf::Zero(7, 1);
                auto H = Eigen::MatrixXf::Identity(7, 7);
                auto kf = new KalmanFilter(7, state, SIGMA_Q, SIGMA_R, A, B, H);
                kf_map[id_num] = kf;
                track_unmatched_cnt_map[id_num] = 0;
                id_num += 1; // 每次创建新track后track id都会+1
            }
        }
        else {
            // 取出所有轨迹的最后一个bbox
            std::vector<BBox> last_bbox_in_each_tracks;
            std::unordered_map<int, uint64_t> track_id_map; // 记录每个track在last_bbox_in_all_tracks中的索引对应的track号
            int idx = 0;
            for(auto &track : all_tracks) {
                last_bbox_in_each_tracks.push_back(track.second.back());
                track_id_map[idx] = track.first;
                idx ++;
            }
            // 构建代价矩阵 m x n
            size_t m = BBoxes.size(), n = last_bbox_in_each_tracks.size();
            std::vector<std::vector<double>> iou_matrix(m, std::vector<double>(n, 0.0f));
            std::vector<std::vector<double>> cost_matrix(m, std::vector<double>(n, 0.0f));
            for(int i = 0; i < m; ++i) {
                for(int j = 0; j < n; ++j) {
                    auto bbox1 = xywh2xyxy(BBoxes[i]);
                    auto bbox2 = xywh2xyxy(last_bbox_in_each_tracks[j]);
                    // https://github.com/mcximing/hungarian-algorithm-cpp 提供的算法不支持负数的代价矩阵
                    // 这里用2 - IoU 而不是用1 - IoU 是为了为以后的DIoU准备
                    iou_matrix[i][j] = bbox_iou(bbox1, bbox2);
                    cost_matrix[i][j] = 2 - iou_matrix[i][j]; 
                }
            }
            // 使用匈牙利算法匹配
            HungarianAlgorithm HungAlgo;
            vector<int> assignment;
            double cost = HungAlgo.Solve(cost_matrix, assignment);
            assignment_debug.clear();
            for (size_t i = 0; i < assignment.size(); i++) {
                assignment_debug.push_back(track_id_map[assignment[i]]);
            }
            

            // 记录哪些轨迹被匹配了, 没被匹配的轨迹会依赖kf更新
            auto last_bbox_in_all_tracks_matched_mask = set<uint64_t>();
            for (auto &item : all_tracks) {
                auto track_id = item.first;
                last_bbox_in_all_tracks_matched_mask.insert(track_id);
            }

            for (size_t i = 0; i < BBoxes.size(); i++) {
                BBox_xyxy bbox = xywh2xyxy(BBoxes[i]);
            }
            
            for (int i = 0; i < assignment.size(); ++i) {
                if(assignment[i] == -1) {
                    // 有未匹配的BBox 则初始化新的 track
                    printf("no match, id=%d\n", i);
                    track_t track;
                    auto &bbox = BBoxes[i];
                    track.push_back(bbox);
                    all_tracks[id_num] = track;

                    auto state = Eigen::VectorXf(7);
                    float x = bbox.x, y = bbox.y, s = bbox.w * bbox.h, r = bbox.w / (bbox.h + 1e-5f);
                    state << x, y, s, r, 0, 0, 0;
                    auto A = Eigen::MatrixXf(7, 7);
                    A << 1, 0, 0, 0, 1, 0, 0,
                         0, 1, 0, 0, 0, 1, 0,
                         0, 0, 1, 0, 0, 0, 1,
                         0, 0, 0, 1, 0, 0, 0,
                         0, 0, 0, 0, 1, 0, 0,
                         0, 0, 0, 0, 0, 1, 0,
                         0, 0, 0, 0, 0, 0, 1;
                    auto B = Eigen::MatrixXf::Zero(7, 1);
                    auto H = Eigen::MatrixXf::Identity(7, 7);
                    auto kf = new KalmanFilter(7, state, SIGMA_Q, SIGMA_R, A, B, H);
                    kf_map[id_num] = kf;
                    track_unmatched_cnt_map[id_num] = 0;
                    id_num += 1; // 每次创建新track后track id都会+1
                }
                else if (iou_matrix[i][assignment[i]] > 0.1f) {
                    uint64_t track_id = track_id_map[assignment[i]];
                    last_bbox_in_all_tracks_matched_mask.erase(track_id); // 匹配上就从未匹配的轨迹集合中移除
                    auto kf = kf_map[track_id];
                    auto z = Eigen::VectorXf(7); // 测量值
                    float x1 = BBoxes[i].x,
                          y1 = BBoxes[i].y,
                          s1 = BBoxes[i].w * BBoxes[i].h,
                          r1 = BBoxes[i].w / (BBoxes[i].h + 1e-5f);
                    float x2 = last_bbox_in_each_tracks[assignment[i]].x, 
                          y2 = last_bbox_in_each_tracks[assignment[i]].y,
                          s2 = last_bbox_in_each_tracks[assignment[i]].w * last_bbox_in_each_tracks[assignment[i]].h,
                          r2 = last_bbox_in_each_tracks[assignment[i]].w / (last_bbox_in_each_tracks[assignment[i]].h + 1e-5f);
                    z << x1, y1, s1, r1, x1 - x2, y1 - y2, s1 - s2;
                    auto x_posterior = kf->predict(z);
                    float x3 = x_posterior(0),
                          y3 = x_posterior(1),
                          s3 = x_posterior(2),
                          r3 = x_posterior(3);
                          s3 = s3 > 0 ? s3 : 1e-5f;
                          r3 = r3 > 0 ? r3 : 1e-5f;
                    float w3 = sqrtf(s3 * r3),
                          h3 = sqrtf(s3 / r3);
                    all_tracks[track_id].push_back(BBox{x3, y3, w3, h3});
                    track_unmatched_cnt_map[track_id] = 0; // 匹配成功则将未匹配计数器清零
                }
            }
            for (auto &unmatched_track_id : last_bbox_in_all_tracks_matched_mask)
            {
                
                if (unmatched_track_id == 4)
                {
                    printf("Debug\n");
                }
                // print all the unmatched tracks
                std::cout << "unmatched track id: " << unmatched_track_id << std::endl;
                auto last_bbox = all_tracks[unmatched_track_id].back();
                auto kf = kf_map[unmatched_track_id];
                kf->x_posterior = kf->A * kf->x_posterior;
                // kf->x_posterior = x_posterior;
                // print kf->x_posterior
                std::cout << "kf->x_posterior: \n" << kf->x_posterior << std::endl;
                std::cout << "---------------------------" << std::endl;
                float x3 = kf->x_posterior(0),
                      y3 = kf->x_posterior(1),
                      s3 = kf->x_posterior(2),
                      r3 = kf->x_posterior(3);
                      s3 = s3 > 0 ? s3 : 1e-5f;
                      r3 = r3 > 0 ? r3 : 1e-5f;
                float w3 = sqrtf(s3 * r3),
                      h3 = sqrtf(s3 / r3);
                all_tracks[unmatched_track_id].push_back(BBox{x3, y3, w3, h3, last_bbox.score, last_bbox.cls});
            }
        }

        // 维护轨迹
        std::vector<uint64_t> id_of_tracks_to_be_removed;
        for (auto &item : all_tracks) {
            auto &track = item.second;
            auto &id = item.first;
            // 维护轨迹长度
            if(track.size() > TRACK_MAX_LEN) {
                track.pop_front();
            }
            // 维护未匹配计数器
            track_unmatched_cnt_map[id] += 1;
            if (track_unmatched_cnt_map[id] > MAX_AGE) {
                id_of_tracks_to_be_removed.push_back(id);
            }
            float x = track.back().x,
                  y = track.back().y,
                  w = track.back().w,
                  h = track.back().h;
            draw_box_in_color(frame, Box{x, y, w, h}, cv::Scalar(255, 255, 255), 2);
        }
        // 删除未匹配计数器超过阈值的轨迹
        for (auto id : id_of_tracks_to_be_removed) {
                all_tracks.erase(id);
                kf_map.erase(id);
                track_unmatched_cnt_map.erase(id);
        }
        
        // 绘制轨迹线
        draw_tracks(frame, all_tracks, 3);

        draw_bboxes(frame, BBoxes, detector.names, 0.8f, 1);
        // 显示帧
        cv::imshow("Frame", frame);
        // 按 'ESC' 键退出循环
        if(cv::waitKey(1) == 27) {
            break;
        }
    }
    return 0;
}
