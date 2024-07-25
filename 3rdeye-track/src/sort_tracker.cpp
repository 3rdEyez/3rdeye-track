#include "sort_tracker.h"

sort_tracker::sort_tracker(float sigma_q, float sigma_r, uint64_t track_max_len, uint64_t max_age)
{
    // id_num 用于标识每个轨迹的id
    id_num = 0;

    this->sigma_q = sigma_q;
    this->sigma_r = sigma_r;
    this->track_max_len = track_max_len;
    this->max_age = max_age;
}

sort_tracker::~sort_tracker()
{
    // auto kf = new KalmanFilter(7, state, this->sigma_q, this->sigma_r, A, B, H);
    // delete kf;
    for (auto item : all_tracks) {
        auto track_id = item.first;
        delete kf_map[track_id];
    }
}

void sort_tracker::exec(const std::vector<BBox>  &bboxes)
{
    std::vector<BBox> new_track_bboxes; // 存储bbox, 用于创建新轨迹
    if (all_tracks.size() == 0) {
        // 如果tracks为空，则初始化
        for(auto &bbox : bboxes) {
            new_track_bboxes.push_back(bbox); // 后续根据new_track_bboxes来创建 track和 kf
        }
    }
    else
    {
        std::vector<BBox> last_bbox_in_each_tracks; // 取出所有轨迹的最后一个bbox
        std::unordered_map<int, uint64_t> track_id_map; // 记录每个track在last_bbox_in_all_tracks中的索引对应的track号
        int idx = 0;
        for(auto &track : all_tracks) {
            last_bbox_in_each_tracks.push_back(track.second.back());
            track_id_map[idx] = track.first;
            idx ++;
        }
        // 构建代价矩阵 m x n
        size_t m = bboxes.size(), n = last_bbox_in_each_tracks.size();
        std::vector<std::vector<double>> iou_matrix(m, std::vector<double>(n, 0.0f));
        std::vector<std::vector<double>> cost_matrix(m, std::vector<double>(n, 0.0f));
        for(int i = 0; i < m; ++i) {
            for(int j = 0; j < n; ++j) {
                auto bbox1 = xywh2xyxy(bboxes[i]);
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
        double cost;
        if (m > 0 && n > 0) cost = HungAlgo.Solve(cost_matrix, assignment);
        // 记录哪些轨迹被匹配了, 没被匹配的轨迹会依赖kf更新
        auto last_bbox_in_all_tracks_matched_mask = set<uint64_t>();
        for (auto &item : all_tracks) {
            auto track_id = item.first;
            last_bbox_in_all_tracks_matched_mask.insert(track_id);
        }
        
        for (int i = 0; i < assignment.size(); ++i) {
            if(assignment[i] == -1 || \
                (assignment[i] != -1 && iou_matrix[i][assignment[i]] < 0.1f)) {
                // 有未匹配的BBox 或者IoU非常低的BBox 则初始化新的 track
                printf("no match, id=%d\n", i);
                track_t track;
                auto &bbox = bboxes[i];
                new_track_bboxes.push_back(bbox); // 后续根据new_track_bboxes来创建 track和 kf
            }
            else if (iou_matrix[i][assignment[i]] > 0.1f) {
                uint64_t track_id = track_id_map[assignment[i]];
                last_bbox_in_all_tracks_matched_mask.erase(track_id); // 匹配上就从未匹配的轨迹集合中移除
                auto kf = kf_map[track_id];
                auto z = Eigen::VectorXf(7); // 测量值
                float x1 = bboxes[i].x,
                        y1 = bboxes[i].y,
                        s1 = bboxes[i].w * bboxes[i].h,
                        r1 = bboxes[i].w / (bboxes[i].h + 1e-5f);
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
            auto last_bbox = all_tracks[unmatched_track_id].back();
            auto kf = kf_map[unmatched_track_id];
            kf->x_posterior = kf->A * kf->x_posterior;
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
    
    for(auto &bbox:new_track_bboxes) {
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
        auto kf = new KalmanFilter(7, state, this->sigma_q, this->sigma_r, A, B, H);
        kf_map[id_num] = kf;
        track_unmatched_cnt_map[id_num] = 0;
        id_num += 1; // 每次创建新track后track id都会+1
    }

    new_track_bboxes.clear(); // 为下一轮新增轨迹做准备

    // 维护轨迹
    std::vector<uint64_t> id_of_tracks_to_be_removed;
    for (auto &item : all_tracks) {
        auto &track = item.second;
        auto &id = item.first;
        // 维护轨迹长度
        if(track.size() > this->track_max_len) {
            track.pop_front();
        }
        // 维护未匹配计数器
        track_unmatched_cnt_map[id] += 1;
        if (track_unmatched_cnt_map[id] > this->max_age) {
            id_of_tracks_to_be_removed.push_back(id);
        }
        float x = track.back().x,
                y = track.back().y,
                w = track.back().w,
                h = track.back().h;
    }
    // 删除未匹配计数器超过阈值的轨迹
    for (auto id : id_of_tracks_to_be_removed) {
            all_tracks.erase(id);
            delete kf_map[id];
            kf_map.erase(id);
            track_unmatched_cnt_map.erase(id);
    }
}

const tracks_t& sort_tracker::get_all_tracks()
{
    return all_tracks;
}