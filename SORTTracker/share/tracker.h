#ifndef __SHARE_TRACKER_H__
#define __SHARE_TRACKER_H__

#include "./dtype.h"
#include <vector>

int tracker_init(float sigma_q, float sigma_r, int track_max_len, int max_age);
void tracker_destroy(int track_id);
bool tracker_update(int track_id, const std::vector<BBox>& bboxes);
const trackmap_t &tracker_get_tracks(int track_id);

#endif // __SHARE_TRACKER_H__
