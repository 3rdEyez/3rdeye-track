#ifndef __SHARE_TRACKER_H__
#define __SHARE_TRACKER_H__

#include <vector>
#include <queue>
#include <unordered_map>

typedef struct {
    float x;
    float y;
    float w;
    float h;
    float score;
    uint32_t cls;
}BBox;


typedef std::deque<BBox> track_t;
typedef std::unordered_map<uint64_t, track_t> trackmap_t;

int tracker_init(float sigma_q, float sigma_r, int track_max_len, int max_age);
void tracker_destroy(int track_id);
bool tracker_update(int track_id, const std::vector<BBox>& bboxes);
const trackmap_t &tracker_get_tracks(int track_id);


#endif // __SHARE_TRACKER_H__
