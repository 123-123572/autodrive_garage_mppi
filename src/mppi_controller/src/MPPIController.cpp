#include "mppi_controller/MPPIController.hpp"
#include <cmath>

#include <omp.h> // 顶部记得加入 OpenMP 头文件

namespace autodrive_garage::mppi {

MPPIController::ptr MPPIController::create(const Config& config, KinematicBicycleModel::ptr model) {
    return std::make_unique<MPPIController>(config, std::move(model));
}

MPPIController::MPPIController(const Config& config, KinematicBicycleModel::ptr model)
    : config_(config), 
      model_(std::move(model)),
      rng_(std::random_device{}()),
      dist_accel_(0.0, config.std_dev_accel),
      dist_steer_(0.0, config.std_dev_steer) 
{
    // 构造函数中一次性分配所有需要的内存空间！
    control_sequence_ = Eigen::MatrixXd::Zero(2, config_.horizon);
    noise_accel_      = Eigen::MatrixXd::Zero(config_.num_samples, config_.horizon);
    noise_steer_      = Eigen::MatrixXd::Zero(config_.num_samples, config_.horizon);
    trajectory_costs_ = Eigen::VectorXd::Zero(config_.num_samples);
}

 // 高斯噪声轨迹生成
void MPPIController::GenerateNoise() {
    for (int k = 0; k < config_.num_samples; ++k) {
        for (int t = 0; t < config_.horizon; ++t) {
            noise_accel_(k, t) = dist_accel_(rng_);
            noise_steer_(k, t) = dist_steer_(rng_);
        }
    }
}

ControlVec MPPIController::ComputeControl(const StateVec& current_state, const Eigen::MatrixXd& reference_trajectory) {
    // 1. 生成所有样本的控制噪声 (当前使用单线程随机数生成，避免 std::mt19937 的线程安全问题)
    GenerateNoise();

    // 2. Rollout: 彻底释放 CPU 多核算力
    // 使用 dynamic 调度，按块分配任务，平衡各线程负载
    #pragma omp parallel for schedule(dynamic, 32)
    for (int k = 0; k < config_.num_samples; ++k) {
        StateVec sim_state = current_state;
        double cost = 0.0;

        for (int t = 0; t < config_.horizon; ++t) {
            // 叠加控制量：当前基准控制序列 + 随机噪声
            ControlVec u;
            u(0) = control_sequence_(0, t) + noise_accel_(k, t);
            u(1) = control_sequence_(1, t) + noise_steer_(k, t);

            // 物理模型推演 (使用你写的极速版 Bicycle Model)
            sim_state = model_->UpdateState(sim_state, u, config_.dt);

            // 累加轨迹代价 (与参考线的横向/纵向误差)
            double dx = sim_state(kinematic_bicycle::X) - reference_trajectory(0, t);
            double dy = sim_state(kinematic_bicycle::Y) - reference_trajectory(1, t);
            
            // 终端代价与过程代价可以根据需求调整权重，目前极简处理
            cost += (dx * dx + dy * dy); 
        }
        trajectory_costs_(k) = cost;
    }

    // 3. 计算权重 (Information Theoretic Weighting) 
    double min_cost = trajectory_costs_.minCoeff(); 
    double sum_weights = 0.0;
    Eigen::VectorXd weights = Eigen::VectorXd::Zero(config_.num_samples);

    // 权重计算也可以并行化，结合 reduction 归约 sum_weights
    #pragma omp parallel for reduction(+:sum_weights)
    for (int k = 0; k < config_.num_samples; ++k) {
        weights(k) = std::exp(-(trajectory_costs_(k) - min_cost) / config_.lambda);
        sum_weights += weights(k);
    }

    // 4. 更新最优控制序列
    for (int t = 0; t < config_.horizon; ++t) {
        double delta_accel = 0.0;
        double delta_steer = 0.0;

        // 这里因为外层是 t 循环，内层 k 循环可以利用 SIMD 指令或简单的 OpenMP 规约
        #pragma omp parallel for reduction(+:delta_accel, delta_steer)
        for (int k = 0; k < config_.num_samples; ++k) {
            delta_accel += weights(k) * noise_accel_(k, t);
            delta_steer += weights(k) * noise_steer_(k, t);
        }

        control_sequence_(0, t) += delta_accel / sum_weights;
        control_sequence_(1, t) += delta_steer / sum_weights;
    }

    // 5. 提取当前时刻 (t=0) 的控制量输出
    ControlVec optimal_control;
    optimal_control(0) = control_sequence_(0, 0);
    optimal_control(1) = control_sequence_(1, 0);

    // 6. 热启动(Warm Start) / 序列左移
    for (int t = 0; t < config_.horizon - 1; ++t) {
        control_sequence_.col(t) = control_sequence_.col(t + 1);
    }
    control_sequence_.col(config_.horizon - 1).setZero();

    return optimal_control;
}

} // namespace autodrive_garage::mppi