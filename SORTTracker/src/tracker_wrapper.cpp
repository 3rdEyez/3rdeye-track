#include "tracker.h"
#include  "sort_tracker.h"

using namespace std;
static std::unordered_map<int, sort_tracker*> trackers;

int tracker_init(float sigma_q, float sigma_r, int track_max_len, int max_age)
{
    static int track_id = 0;
    trackers[track_id] = new sort_tracker(sigma_q, sigma_r, (uint64_t)track_max_len, (uint64_t)max_age);
    return track_id++;
}

void tracker_destroy(int track_id)
{
    delete trackers[track_id];
    trackers.erase(track_id);
}

bool tracker_update(int track_id, const std::vector<BBox>& bboxes)
{
    if (!trackers.count(track_id)) return false;
    trackers[track_id]->exec(bboxes);
    return true;
}
const trackmap_t &tracker_get_tracks(int track_id)
{
    return trackers[track_id]->get_all_tracks();
}