#pragma once

#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>

#include "mppi_controller/hybrid_a_star.hpp"

namespace autodrive_garage {

class HybridAStarNode : public rclcpp::Node {
public:
    HybridAStarNode();
    ~HybridAStarNode() = default;

private:
    // 回调函数
    void MapCallback(const nav_msgs::msg::OccupancyGrid::SharedPtr msg);
    void OdomCallback(const nav_msgs::msg::Odometry::SharedPtr msg);
    void GoalCallback(const geometry_msgs::msg::PoseStamped::SharedPtr goal_msg);
    
    // 内部方法
    void PublishPath(const std::vector<planning::HybridAStarNode::ptr>& plan_nodes);

    // 成员变量
    planning::HybridAStar::ptr planner_;

    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr goal_sub_;
    rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr map_sub_;

    nav_msgs::msg::Odometry::SharedPtr current_odom_;
    bool map_received_{false};
};

} // namespace autodrive_garage