#ifndef __SORT_TRACKER_H__
#define __SORT_TRACKER_H__

#include "detect_utils.h"
#include "kalman_filter.h"
#include "Hungarian.h"
#include <set>
#include <unordered_map>


typedef std::deque<BBox> track_t;
typedef std::unordered_map<uint64_t, track_t> trackmap_t;


class sort_tracker
{
public:
    sort_tracker(float sigma_q, float sigma_r, uint64_t track_max_len, uint64_t max_age);
    ~sort_tracker();
    void exec(const std::vector<BBox>  &bboxes);
    const trackmap_t &get_all_tracks();

private:
    uint64_t id_num;
    trackmap_t all_tracks;
    std::unordered_map<uint64_t, KalmanFilter*> kf_map; // 存储每个track的KalmanFilter
    std::unordered_map<uint64_t, uint64_t> track_unmatched_cnt_map; // 存储每个track的未匹配帧数
    float sigma_q, sigma_r;
    uint64_t track_max_len, max_age;
};

#endif

