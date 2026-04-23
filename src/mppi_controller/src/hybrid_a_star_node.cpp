#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <tf2/utils.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <chrono>

#include "mppi_controller/hybrid_a_star.hpp" // 包含你写的规划器头文件

using namespace autodrive_garage;

class HybridAStarNode : public rclcpp::Node {
public:
    HybridAStarNode() : Node("hybrid_a_star_node") {
        RCLCPP_INFO(this->get_logger(), "🧠 Hybrid A* Global Planner Node 正在启动...");

        // 1. 初始化规划器配置 (这里可以后续替换为从 ROS 2 参数服务器读取)
        planning::HybridAStarConfig config;
        config.xy_resolution = 0.2;
        config.theta_resolution = 0.1;
        config.step_size = 0.5;
        planner_ = planning::HybridAStar::create(config);

        // 2. 初始化 ROS 接口
        // 发布全局参考路径，喂给 MPPI
        path_pub_ = this->create_publisher<nav_msgs::msg::Path>("/reference_path", 10);

        // 订阅里程计 (获取车辆当前位置作为起点)
        odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
            "/odom", 10, std::bind(&HybridAStarNode::OdomCallback, this, std::placeholders::_1));

        // 订阅 RViz 的 2D Nav Goal (获取目标终点)
        goal_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
            "/goal_pose", 10, std::bind(&HybridAStarNode::GoalCallback, this, std::placeholders::_1));

        RCLCPP_INFO(this->get_logger(), "✅ 规划器就绪！请在 RViz 中使用 '2D Nav Goal' 下发目标点...");
    }

private:
    planning::HybridAStar::ptr planner_;

    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr goal_sub_;

    nav_msgs::msg::Odometry::SharedPtr current_odom_;

    // 缓存当前位姿
    void OdomCallback(const nav_msgs::msg::Odometry::SharedPtr msg) {
        current_odom_ = msg;
    }

    // 接收到目标点时，触发一次全局规划！
    void GoalCallback(const geometry_msgs::msg::PoseStamped::SharedPtr goal_msg) {
        if (!current_odom_) {
            RCLCPP_WARN(this->get_logger(), "❌ 尚未接收到 Odom 数据，无法规划！");
            return;
        }

        // 1. 提取起点状态 (来自 /odom)
        double start_x = current_odom_->pose.pose.position.x;
        double start_y = current_odom_->pose.pose.position.y;
        double start_theta = tf2::getYaw(current_odom_->pose.pose.orientation);

        // 2. 提取终点状态 (来自 /goal_pose)
        double goal_x = goal_msg->pose.position.x;
        double goal_y = goal_msg->pose.position.y;
        double goal_theta = tf2::getYaw(goal_msg->pose.orientation);

        RCLCPP_INFO(this->get_logger(), "📍 收到新任务: 起点(%.2f, %.2f) -> 终点(%.2f, %.2f)", 
                    start_x, start_y, goal_x, goal_y);

        // 3. 准备接收路径
        std::vector<planning::HybridAStarNode::ptr> result_path;

        // ==================== 🚀 性能打桩开始 ====================
        auto start_time = std::chrono::high_resolution_clock::now();

        // 🧠 调用你的 Hybrid A* 核心算法
        bool success = planner_->Plan(start_x, start_y, start_theta, 
                                      goal_x, goal_y, goal_theta, 
                                      result_path);

        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = end_time - start_time;
        // ==================== 🚀 性能打桩结束 ====================

        if (success) {
            RCLCPP_INFO(this->get_logger(), "🟢 规划成功！耗时: %.2f ms | 路径点数: %zu", 
                        elapsed.count(), result_path.size());
            PublishPath(result_path);
        } else {
            RCLCPP_ERROR(this->get_logger(), "🔴 规划失败！耗时: %.2f ms", elapsed.count());
        }
    }

    // 将你自定义的路径节点转换为 ROS 2 标准消息并发布
    void PublishPath(const std::vector<planning::HybridAStarNode::ptr>& plan_nodes) {
        nav_msgs::msg::Path path_msg;
        path_msg.header.stamp = this->get_clock()->now();
        path_msg.header.frame_id = "odom"; // 或者 "map"，取决于你的坐标系设定

        for (const auto& node : plan_nodes) {
            geometry_msgs::msg::PoseStamped pose;
            pose.header = path_msg.header;
            
            // 还原连续坐标
            pose.pose.position.x = node->x;
            pose.pose.position.y = node->y;
            pose.pose.position.z = 0.0;

            // 航向角转四元数
            tf2::Quaternion q;
            q.setRPY(0, 0, node->theta);
            pose.pose.orientation = tf2::toMsg(q);

            path_msg.poses.push_back(pose);
        }

        path_pub_->publish(path_msg);
    }
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<HybridAStarNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}