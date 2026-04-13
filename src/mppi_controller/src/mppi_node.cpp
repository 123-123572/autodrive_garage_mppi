#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <tf2/utils.h> // 注意这里一般是 .h 不是 .hpp
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <chrono>

#include "mppi_controller/MPPIController.hpp"

using namespace autodrive_garage;

class MPPIControlNode : public rclcpp::Node {
public:
    MPPIControlNode() : Node("mppi_control_node") {
        RCLCPP_INFO(this->get_logger(), "🚀 MPPI Control Node 正在启动...");

        kinematic_bicycle::KinematicBicycleModel::Config model_cfg;
        auto model = kinematic_bicycle::KinematicBicycleModel::create(model_cfg);
        
        mppi::MPPIController::Config mppi_cfg;
        mppi_cfg.num_samples = 1000; // 明确标出样本数，方便后续对比
        mppi_cfg.horizon = 50;
        mppi_cfg.lambda = 2.0; 
        mppi_ = mppi::MPPIController::create(mppi_cfg, std::move(model));

        cmd_pub_ = this->create_publisher<geometry_msgs::msg::Twist>("/cmd_vel", 10);

        odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
            "/odom", 10, std::bind(&MPPIControlNode::OdomCallback, this, std::placeholders::_1));
        
        path_sub_ = this->create_subscription<nav_msgs::msg::Path>(
            "/reference_path", 10, std::bind(&MPPIControlNode::PathCallback, this, std::placeholders::_1));

        control_timer_ = this->create_wall_timer(
            std::chrono::milliseconds(20), std::bind(&MPPIControlNode::TimerCallback, this));

        RCLCPP_INFO(this->get_logger(), "✅ MPPI 初始化完成，等待数据接入...");
    }

private:
    mppi::MPPIController::ptr mppi_;
    
    rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr cmd_pub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
    rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr path_sub_;
    rclcpp::TimerBase::SharedPtr control_timer_;

    nav_msgs::msg::Odometry::SharedPtr current_odom_;
    nav_msgs::msg::Path::SharedPtr current_path_;

    void OdomCallback(const nav_msgs::msg::Odometry::SharedPtr msg) { current_odom_ = msg; }
    void PathCallback(const nav_msgs::msg::Path::SharedPtr msg) { current_path_ = msg; }

    void TimerCallback() {
        if (!current_odom_ || !current_path_) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000, "等待 Odom 和 Path 数据...");
            return;
        }

        // 1. Odom -> Eigen::Vector4d
        mppi::StateVec current_state;
        current_state(kinematic_bicycle::X) = current_odom_->pose.pose.position.x;
        current_state(kinematic_bicycle::Y) = current_odom_->pose.pose.position.y;
        current_state(kinematic_bicycle::YAW) = tf2::getYaw(current_odom_->pose.pose.orientation);
        current_state(kinematic_bicycle::V) = current_odom_->twist.twist.linear.x;

        // 2. Path -> Eigen::MatrixXd
        int horizon = 50; 
        Eigen::MatrixXd ref_traj = Eigen::MatrixXd::Zero(2, horizon);
        int path_size = current_path_->poses.size();

        for (int t = 0; t < horizon; ++t) {
            if (t < path_size) {
                ref_traj(0, t) = current_path_->poses[t].pose.position.x;
                ref_traj(1, t) = current_path_->poses[t].pose.position.y;
            } else if (path_size > 0) {
                ref_traj(0, t) = current_path_->poses.back().pose.position.x;
                ref_traj(1, t) = current_path_->poses.back().pose.position.y;
            }
        }

        // ==================== 🚀 性能打桩开始 ====================
        auto start = std::chrono::high_resolution_clock::now();

        // 3. 呼叫大脑！调用 MPPI 核心算法
        mppi::ControlVec optimal_u = mppi_->ComputeControl(current_state, ref_traj);

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = end - start;
        
        // 性能数据统计与打印 (每 100 次循环输出一次)
        static double total_ms = 0.0;
        static int count = 0;
        total_ms += elapsed.count();
        count++;

        if (count % 100 == 0) {
            RCLCPP_INFO(this->get_logger(), 
                "📊 CPU MPPI 性能测试 | 样本数: 1000 | 平均推演耗时: %.3f ms", 
                total_ms / 100.0);
            total_ms = 0.0;
            count = 0;
        }
        // ==================== 🚀 性能打桩结束 ====================

        // 4. 解析结果并发给底盘
        geometry_msgs::msg::Twist cmd_msg;
        double dt = 0.02; 
        double target_v = current_state(kinematic_bicycle::V) + optimal_u(kinematic_bicycle::ACCEL) * dt;
        cmd_msg.linear.x = target_v;

        double wheelbase = 2.8; 
        double target_yaw_rate = (target_v / wheelbase) * std::tan(optimal_u(kinematic_bicycle::STEER));
        cmd_msg.angular.z = target_yaw_rate;

        cmd_pub_->publish(cmd_msg);
    }
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<MPPIControlNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}