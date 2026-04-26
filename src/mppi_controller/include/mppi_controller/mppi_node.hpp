#pragma once

#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <geometry_msgs/msg/twist.hpp>

#include "mppi_controller/MPPIController.hpp"
#include "mppi_controller/trajectory_processor.hpp"

namespace autodrive_garage {

class MPPIControlNode : public rclcpp::Node {
public:
    MPPIControlNode();
    ~MPPIControlNode() = default;

private:
    // 回调函数
    void OdomCallback(const nav_msgs::msg::Odometry::SharedPtr msg);
    void PathCallback(const nav_msgs::msg::Path::SharedPtr msg);
    void TimerCallback();

    // 核心组件
    mppi::MPPIController::ptr mppi_;
    planning::TrajectoryProcessor::ptr path_processor_;
    
    // ROS 通信
    rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr cmd_pub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
    rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr path_sub_;
    rclcpp::TimerBase::SharedPtr control_timer_;

    // 状态缓存
    nav_msgs::msg::Odometry::SharedPtr current_odom_;
    std::vector<planning::HybridAStarNode::ptr> global_path_nodes_; 

    // 车辆与控制参数
    int horizon_{50};
    double wheelbase_{2.8};
    double dt_{0.02};
};

} // namespace autodrive_garage