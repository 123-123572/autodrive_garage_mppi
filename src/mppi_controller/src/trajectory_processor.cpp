#include "mppi_controller/trajectory_processor.hpp"
#include <cfloat>
#include <algorithm>

namespace autodrive_garage::planning {

TrajectoryProcessor::TrajectoryProcessor(int horizon) 
    : horizon_(horizon), last_nearest_idx_(0) {}

Eigen::MatrixXd TrajectoryProcessor::ExtractLocalRoute(
    double current_x, double current_y, 
    const std::vector<HybridAStarNode::ptr>& global_path) {
    
    if (global_path.empty()) return Eigen::MatrixXd::Zero(2, horizon_);

    last_nearest_idx_ = FindNearestIndex(current_x, current_y, global_path);
    Eigen::MatrixXd local_traj = Eigen::MatrixXd::Zero(2, horizon_);
    int path_size = static_cast<int>(global_path.size());

    for (int i = 0; i < horizon_; ++i) {
        int target_idx = std::min(last_nearest_idx_ + i, path_size - 1);
        local_traj(0, i) = global_path[target_idx]->x;
        local_traj(1, i) = global_path[target_idx]->y;
    }
    return local_traj;
}

int TrajectoryProcessor::FindNearestIndex(
    double x, double y, const std::vector<HybridAStarNode::ptr>& path) {
    
    int best_idx = last_nearest_idx_;
    double min_dist_sq = DBL_MAX;
    int search_range = 100; 
    int end_search = std::min(last_nearest_idx_ + search_range, static_cast<int>(path.size()));

    for (int i = last_nearest_idx_; i < end_search; ++i) {
        double dx = x - path[i]->x;
        double dy = y - path[i]->y;
        double dist_sq = dx * dx + dy * dy;

        if (dist_sq < min_dist_sq) {
            min_dist_sq = dist_sq;
            best_idx = i;
        } else if (dist_sq > min_dist_sq + 1.0) {
            break; 
        }
    }
    return best_idx;
}

} // namespace autodrive_garage::planning