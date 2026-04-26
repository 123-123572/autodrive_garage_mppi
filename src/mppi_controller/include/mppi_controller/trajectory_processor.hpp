#pragma once
#include <Eigen/Dense>
#include <vector>
#include <memory>
#include "mppi_controller/hybrid_a_star.hpp" 

namespace autodrive_garage::planning {

class TrajectoryProcessor {
public:
    using ptr = std::unique_ptr<TrajectoryProcessor>;
    explicit TrajectoryProcessor(int horizon);

    Eigen::MatrixXd ExtractLocalRoute(double current_x, double current_y, 
                                     const std::vector<HybridAStarNode::ptr>& global_path);

private:
    int horizon_;
    int last_nearest_idx_;
    int FindNearestIndex(double x, double y, const std::vector<HybridAStarNode::ptr>& path);
};

} // namespace autodrive_garage::planning