#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <tf2/utils.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>


#include "mppi_controller/MPPIController.hpp"

using namespace autodrive_garage;

class MPPIControlNode : public rclcpp::Node {
public:
    MPPIControlNode() : Node("mppi_control_node") {
        RCLCPP_INFO(this->get_logger(), "🚀 MPPI Control Node 正在启动...");

        // 1. 初始化底层的纯 C++ 控制器 (先用默认配置)
        kinematic_bicycle::KinematicBicycleModel::Config model_cfg;
        auto model = kinematic_bicycle::KinematicBicycleModel::create(model_cfg);
        
        mppi::MPPIController::Config mppi_cfg;
        mppi_cfg.lambda = 2.0; 
        mppi_ = mppi::MPPIController::create(mppi_cfg, std::move(model));

        // 2. 创建发布者 (发给底盘：线速度和角速度)
        cmd_pub_ = this->create_publisher<geometry_msgs::msg::Twist>("/cmd_vel", 10);

        // 3. 创建订阅者 (接收当前车辆位姿，接收 A* 规划的路径)
        odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
            "/odom", 10, std::bind(&MPPIControlNode::OdomCallback, 
                        this, 
                        std::placeholders::_1));
        
        path_sub_ = this->create_subscription<nav_msgs::msg::Path>(
            "/reference_path", 10, std::bind(&MPPIControlNode::PathCallback, 
                        this, 
                        std::placeholders::_1));

        // 4. 创建控制定时器 (控制频率：50Hz -> 20ms 一次)
        control_timer_ = this->create_wall_timer(
            std::chrono::milliseconds(20), 
            std::bind(&MPPIControlNode::TimerCallback, this));

        RCLCPP_INFO(this->get_logger(), "✅ MPPI 初始化完成，等待数据接入...");
    }

private:
    mppi::MPPIController::ptr mppi_;
    
    // ROS 通信接口
    rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr cmd_pub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
    rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr path_sub_;
    rclcpp::TimerBase::SharedPtr control_timer_;

    // 缓存最新数据
    nav_msgs::msg::Odometry::SharedPtr current_odom_;
    nav_msgs::msg::Path::SharedPtr current_path_;

    void OdomCallback(const nav_msgs::msg::Odometry::SharedPtr msg) {
        current_odom_ = msg; // 缓存最新位姿
    }

    void PathCallback(const nav_msgs::msg::Path::SharedPtr msg) {
        current_path_ = msg; // 缓存最新参考路径
    }

    void TimerCallback() {
        // 核心控制大循环：如果没有数据，就待命
        if (!current_odom_ || !current_path_) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000, "等待 Odom 和 Path 数据...");
            return;
        }

        // ==================== 重头戏开始 ====================

        // 1. Odom -> Eigen::Vector4d (X, Y, YAW, V)
        mppi::StateVec current_state;
        current_state(kinematic_bicycle::X) = current_odom_->pose.pose.position.x;
        current_state(kinematic_bicycle::Y) = current_odom_->pose.pose.position.y;
        
        // 四元数转 Yaw 角 (这是 ROS 里极其常用的神技)
        current_state(kinematic_bicycle::YAW) = tf2::getYaw(current_odom_->pose.pose.orientation);
        
        // 读取底盘反馈的当前线速度
        current_state(kinematic_bicycle::V) = current_odom_->twist.twist.linear.x;

        // 2. Path -> Eigen::MatrixXd (提取未来 horizon 个参考点)
        int horizon = 50; // 这里暂时写死，或者你可以给 MPPIController 加个 GetConfig() 方法
        Eigen::MatrixXd ref_traj = Eigen::MatrixXd::Zero(2, horizon);
        int path_size = current_path_->poses.size();

        for (int t = 0; t < horizon; ++t) {
            if (t < path_size) {
                // 如果参考路径还有点，就按顺序取
                ref_traj(0, t) = current_path_->poses[t].pose.position.x;
                ref_traj(1, t) = current_path_->poses[t].pose.position.y;
            } else if (path_size > 0) {
                // 如果 A* 给的路径太短了（快到终点了），就用最后一个点把剩下的 horizon 填满（原地悬停）
                ref_traj(0, t) = current_path_->poses.back().pose.position.x;
                ref_traj(1, t) = current_path_->poses.back().pose.position.y;
            }
        }

        // 3. 呼叫大脑！调用 MPPI 核心算法
        // 这行代码执行完，1000 条轨迹就已经推演完了
        mppi::ControlVec optimal_u = mppi_->ComputeControl(current_state, ref_traj);

        // 4. 解析结果并发给底盘 (ControlVec -> Twist)
        geometry_msgs::msg::Twist cmd_msg;

        // ⚠️ 工程注意点：MPPI 输出的是【加速度(Accel)】，但 Twist 通常接收【目标速度】
        // 我们需要用最基础的物理公式做一层转换：v_target = v_current + a * dt
        double dt = 0.02; // 你的控制频率是 50Hz，所以 dt = 20ms
        double target_v = current_state(kinematic_bicycle::V) + optimal_u(kinematic_bicycle::ACCEL) * dt;
        cmd_msg.linear.x = target_v;

        // ⚠️ 工程注意点 2：MPPI 输出的是【前轮转角(Steer)】，而 Twist 一般接收【角速度(Yaw Rate)】
        // 我们利用自行车模型公式：omega = (v / L) * tan(delta)
        double wheelbase = 2.8; // 轴距
        double target_yaw_rate = (target_v / wheelbase) * std::tan(optimal_u(kinematic_bicycle::STEER));
        cmd_msg.angular.z = target_yaw_rate;

        // 发射！
        cmd_pub_->publish(cmd_msg);

        // ==================== 重头戏结束 ====================
        
        RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 1000, 
            "⚡ 控制下发 -> 目标速度: %.2f m/s, 目标角速度: %.2f rad/s", 
            target_v, target_yaw_rate);
    }
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<MPPIControlNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}