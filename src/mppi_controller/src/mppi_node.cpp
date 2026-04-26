#include "mppi_controller/mppi_node.hpp"

#include <tf2/utils.h> 
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <chrono>
#include <algorithm>
#include <nvtx3/nvToolsExt.h> 

namespace autodrive_garage {

MPPIControlNode::MPPIControlNode() : Node("mppi_control_node") {
    RCLCPP_INFO(this->get_logger(), "🚀 MPPI Control Node 正在启动...");

    // 1. 初始化模型与控制器参数
    kinematic_bicycle::KinematicBicycleModel::Config model_cfg;
    model_cfg.wheelbase = wheelbase_; 
    auto model = kinematic_bicycle::KinematicBicycleModel::create(model_cfg);
    
    mppi::MPPIController::Config mppi_cfg;
    mppi_cfg.num_samples = 20000; 
    mppi_cfg.horizon = horizon_;
    mppi_cfg.lambda = 1.0; 
    mppi_cfg.dt = dt_; 

    mppi_ = mppi::MPPIController::create(mppi_cfg, std::move(model));
    path_processor_ = std::make_unique<planning::TrajectoryProcessor>(horizon_);

    // 2. ROS 接口
    cmd_pub_ = this->create_publisher<geometry_msgs::msg::Twist>("/cmd_vel", 10);
    odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
        "/odom", 10, std::bind(&MPPIControlNode::OdomCallback, this, std::placeholders::_1));
    path_sub_ = this->create_subscription<nav_msgs::msg::Path>(
        "/reference_path", 10, std::bind(&MPPIControlNode::PathCallback, this, std::placeholders::_1));

    int timer_ms = static_cast<int>(dt_ * 1000);
    control_timer_ = this->create_wall_timer(
        std::chrono::milliseconds(timer_ms), std::bind(&MPPIControlNode::TimerCallback, this));

    RCLCPP_INFO(this->get_logger(), "✅ MPPI 控制循环就绪，执行频率: %d Hz", 1000 / timer_ms);
}

void MPPIControlNode::OdomCallback(const nav_msgs::msg::Odometry::SharedPtr msg) { 
    current_odom_ = msg; 
}

void MPPIControlNode::PathCallback(const nav_msgs::msg::Path::SharedPtr msg) { 
    global_path_nodes_.clear();
    for (const auto& pose : msg->poses) {
        auto node = std::make_shared<planning::HybridAStarNode>();
        node->x = pose.pose.position.x;
        node->y = pose.pose.position.y;
        global_path_nodes_.push_back(node);
    }
    RCLCPP_INFO(this->get_logger(), "📥 收到新全局路径，节点数: %zu", global_path_nodes_.size());
}

void MPPIControlNode::TimerCallback() {
    if (!current_odom_ || global_path_nodes_.empty()) return;

    nvtxRangePushA("MPPI_Loop");

    mppi::StateVec current_state;
    current_state(kinematic_bicycle::X) = current_odom_->pose.pose.position.x;
    current_state(kinematic_bicycle::Y) = current_odom_->pose.pose.position.y;
    current_state(kinematic_bicycle::YAW) = tf2::getYaw(current_odom_->pose.pose.orientation);
    current_state(kinematic_bicycle::V) = current_odom_->twist.twist.linear.x;

    // 2. 局部路径提取 (彻底绕过黑盒，100% 内存与逻辑安全版)
        nvtxRangePushA("Path_Extraction");
        Eigen::MatrixXd ref_traj = Eigen::MatrixXd::Zero(2, horizon_);

        if (global_path_nodes_.size() < 2) {
            // 极度安全机制：如果没有足够的路径，直接锚定在车身当前位置原地悬停
            for (int t = 0; t < horizon_; ++t) {
                ref_traj(0, t) = current_state(kinematic_bicycle::X);
                ref_traj(1, t) = current_state(kinematic_bicycle::Y);
            }
        } else {
            // 第一步：全局遍历寻找距离当前车体最近的路径点 (每次都从头找，不怕换新路径)
            int closest_idx = 0;
            double min_dist = 1e9;
            double cur_x = current_state(kinematic_bicycle::X);
            double cur_y = current_state(kinematic_bicycle::Y);

            for (size_t i = 0; i < global_path_nodes_.size(); ++i) {
                double dx = global_path_nodes_[i]->x - cur_x;
                double dy = global_path_nodes_[i]->y - cur_y;
                double dist = dx * dx + dy * dy;
                if (dist < min_dist) {
                    min_dist = dist;
                    closest_idx = i;
                }
            }

            // 第二步：从最近点开始，严格往后截取 horizon_ (50) 个点
            for (int t = 0; t < horizon_; ++t) {
                int target_idx = closest_idx + t;
                
                // 第三步：防越界装甲。如果推演超出了终点，就原地复制最后一个点补齐
                if (target_idx >= static_cast<int>(global_path_nodes_.size())) {
                    target_idx = global_path_nodes_.size() - 1;
                }
                
                // 绝对安全的赋值操作
                ref_traj(0, t) = global_path_nodes_[target_idx]->x;
                ref_traj(1, t) = global_path_nodes_[target_idx]->y;
            }
        }
        nvtxRangePop();

    auto start = std::chrono::high_resolution_clock::now();
    
    nvtxRangePushA("MPPI_Compute");
    mppi::ControlVec optimal_u = mppi_->ComputeControl(current_state, ref_traj);
    nvtxRangePop();

    if (ref_traj.cols() == 0) {
            return; // 提取失败，车辆保持当前指令或停车
        }
        
        // 如果提取到的参考轨迹不够 MPPI 的 horizon 长度，用最后一个点补齐
        if (ref_traj.cols() < horizon_) {
            Eigen::MatrixXd padded_traj(2, horizon_);
            int actual_len = ref_traj.cols();
            
            // 拷贝已有的点
            padded_traj.leftCols(actual_len) = ref_traj;
            
            // 剩余的步数全部原地驻留（目标点悬停）
            for (int t = actual_len; t < horizon_; ++t) {
                padded_traj.col(t) = ref_traj.col(actual_len - 1);
            }
            ref_traj = padded_traj;    

    auto end = std::chrono::high_resolution_clock::now();
    
    geometry_msgs::msg::Twist cmd_msg;
    double target_v = current_state(kinematic_bicycle::V) + optimal_u(kinematic_bicycle::ACCEL) * dt_;
    target_v = std::max(0.0, std::min(target_v, 15.0)); 
    cmd_msg.linear.x = target_v;

    if (target_v > 0.1) {
        cmd_msg.angular.z = (target_v / wheelbase_) * std::tan(optimal_u(kinematic_bicycle::STEER));
    } else {
        cmd_msg.angular.z = 0.0;
    }

    cmd_pub_->publish(cmd_msg);

    nvtxRangePop(); 

    static double total_ms = 0.0;
    static int count = 0;
    total_ms += std::chrono::duration<double, std::milli>(end - start).count();
    if (++count % 50 == 0) { 
        RCLCPP_INFO(this->get_logger(), "⚡ GPU 核心推演耗时: %.3f ms | 当前车速: %.2f m/s", total_ms / 50.0, target_v);
        total_ms = 0.0; count = 0;
    }
    }
}

} // namespace autodrive_garage

// ================= 主函数入口 =================
int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<autodrive_garage::MPPIControlNode>());
    rclcpp::shutdown();
    return 0;
}