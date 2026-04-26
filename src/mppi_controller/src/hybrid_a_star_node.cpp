#include "mppi_controller/hybrid_a_star_node.hpp"

#include <tf2/utils.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <chrono>

namespace autodrive_garage {

HybridAStarNode::HybridAStarNode() : Node("hybrid_a_star_node") {
    RCLCPP_INFO(this->get_logger(), "🧠 Hybrid A* Global Planner Node 正在启动...");

    planning::HybridAStarConfig config;
    config.xy_resolution = 0.2;
    config.theta_resolution = 0.1;
    config.step_size = 0.5;
    planner_ = planning::HybridAStar::create(config);

    path_pub_ = this->create_publisher<nav_msgs::msg::Path>("/reference_path", 10);

    odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
        "/odom", 10, std::bind(&HybridAStarNode::OdomCallback, this, std::placeholders::_1));

    goal_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
        "/goal_pose", 10, std::bind(&HybridAStarNode::GoalCallback, this, std::placeholders::_1));

    map_sub_ = this->create_subscription<nav_msgs::msg::OccupancyGrid>(
        "/map", 1, std::bind(&HybridAStarNode::MapCallback, this, std::placeholders::_1));

    RCLCPP_INFO(this->get_logger(), "✅ 规划器就绪！等待地图与位姿数据...");
}

void HybridAStarNode::MapCallback(const nav_msgs::msg::OccupancyGrid::SharedPtr msg) {
    int width = msg->info.width;
    int height = msg->info.height;

    double origin_x = msg->info.origin.position.x;
    double origin_y = msg->info.origin.position.y;
    
    std::vector<uint8_t> processed_costmap(msg->data.size(), 0);

    for (size_t i = 0; i < msg->data.size(); ++i) {
        int8_t cost = msg->data[i];
        if (cost < 0) {
            processed_costmap[i] = 255; 
        } else {
            processed_costmap[i] = static_cast<uint8_t>(cost * 2.55); 
        }
    }

   planner_->UpdateMap(processed_costmap, width, height, origin_x, origin_y);

    if (!map_received_) {
        RCLCPP_INFO(this->get_logger(), "🗺️ 成功加载全局代价地图! (%d x %d)", width, height);
        map_received_ = true;
    }
}

void HybridAStarNode::OdomCallback(const nav_msgs::msg::Odometry::SharedPtr msg) {
    current_odom_ = msg;
}

void HybridAStarNode::GoalCallback(const geometry_msgs::msg::PoseStamped::SharedPtr goal_msg) {
    if (!current_odom_ || !map_received_) {
        RCLCPP_WARN(this->get_logger(), "❌ 缺少 Odom 或 Map 数据，无法规划！");
        return;
    }

    double start_x = current_odom_->pose.pose.position.x;
    double start_y = current_odom_->pose.pose.position.y;
    double start_theta = tf2::getYaw(current_odom_->pose.pose.orientation);

    double goal_x = goal_msg->pose.position.x;
    double goal_y = goal_msg->pose.position.y;
    double goal_theta = tf2::getYaw(goal_msg->pose.orientation);

    RCLCPP_INFO(this->get_logger(), "📍 开始规划: 起点(%.2f, %.2f) -> 终点(%.2f, %.2f)", 
                start_x, start_y, goal_x, goal_y);

    std::vector<planning::HybridAStarNode::ptr> result_path;

    auto start_time = std::chrono::high_resolution_clock::now();
    bool success = planner_->Plan(start_x, start_y, start_theta, goal_x, goal_y, goal_theta, result_path);
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end_time - start_time;

    if (success) {
        RCLCPP_INFO(this->get_logger(), "🟢 规划成功！耗时: %.2f ms | 路径点数: %zu", elapsed.count(), result_path.size());
        PublishPath(result_path);
    } else {
        RCLCPP_ERROR(this->get_logger(), "🔴 规划失败！耗时: %.2f ms", elapsed.count());
    }
}

void HybridAStarNode::PublishPath(const std::vector<planning::HybridAStarNode::ptr>& plan_nodes) {
    nav_msgs::msg::Path path_msg;
    path_msg.header.stamp = this->get_clock()->now();
    path_msg.header.frame_id = "map"; 

    for (const auto& node : plan_nodes) {
        geometry_msgs::msg::PoseStamped pose;
        pose.header = path_msg.header;
        pose.pose.position.x = node->x;
        pose.pose.position.y = node->y;
        pose.pose.position.z = 0.0;

        tf2::Quaternion q;
        q.setRPY(0, 0, node->theta);
        pose.pose.orientation = tf2::toMsg(q);

        path_msg.poses.push_back(pose);
    }
    path_pub_->publish(path_msg);
}

} // namespace autodrive_garage

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<autodrive_garage::HybridAStarNode>());
    rclcpp::shutdown();
    return 0;
}