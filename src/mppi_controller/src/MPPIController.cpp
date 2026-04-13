#include "mppi_controller/MPPIController.hpp"
#include <cmath>
#include <fstream>

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
    // 1. 生成所有样本的控制噪声
    GenerateNoise();
    std::ofstream debug_csv("mppi_data.csv", std::ios::app);

    // 2. Rollout: 并行前向推演（这里是 Day 6 接入 OpenMP/CUDA 的绝佳切入点）
    // #pragma omp parallel for (后续优化点)
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

            if (k < 1000) {
                debug_csv << "sample," << k << "," << t << "," 
                          << sim_state(kinematic_bicycle::X) << "," 
                          << sim_state(kinematic_bicycle::Y) << "\n";
            }

            // 累加轨迹代价 (与参考线的横向/纵向误差)
            double dx = sim_state(kinematic_bicycle::X) - reference_trajectory(0, t);
            double dy = sim_state(kinematic_bicycle::Y) - reference_trajectory(1, t);
            cost += (dx * dx + dy * dy); 
        }
        trajectory_costs_(k) = cost;
    }

    // 3. 计算权重 (Information Theoretic Weighting) 
    double min_cost = trajectory_costs_.minCoeff(); // 提取最小值用于数值稳定
    double sum_weights = 0.0;
    Eigen::VectorXd weights = Eigen::VectorXd::Zero(config_.num_samples);

    for (int k = 0; k < config_.num_samples; ++k) {
        // Log-Sum-Exp 技巧防止 exp 运算下溢
        weights(k) = std::exp(-(trajectory_costs_(k) - min_cost) / config_.lambda);
        sum_weights += weights(k);
    }

    // 4. 更新最优控制序列
    for (int t = 0; t < config_.horizon; ++t) {
        double delta_accel = 0.0;
        double delta_steer = 0.0;

        for (int k = 0; k < config_.num_samples; ++k) {
            delta_accel += weights(k) * noise_accel_(k, t);
            delta_steer += weights(k) * noise_steer_(k, t);
        }

        // 最终加权平均，并更新基准序列
        control_sequence_(0, t) += delta_accel / sum_weights;
        control_sequence_(1, t) += delta_steer / sum_weights;

    }

    StateVec opt_state = current_state;
    for (int t = 0; t < config_.horizon; ++t) {
    ControlVec opt_u;
    opt_u(0) = control_sequence_(0, t);
    opt_u(1) = control_sequence_(1, t);
        
    // 物理模型推演红线
    opt_state = model_->UpdateState(opt_state, opt_u, config_.dt);
        
    // 写入 CSV
    debug_csv << "opt,0," << t << "," 
              << opt_state(kinematic_bicycle::X) << "," 
              << opt_state(kinematic_bicycle::Y) << "\n";
    }
    debug_csv.close();

    // 5. 提取当前时刻 (t=0) 的控制量输出
    ControlVec optimal_control;
    optimal_control(0) = control_sequence_(0, 0);
    optimal_control(1) = control_sequence_(1, 0);

    

    // 6. 热启动(Warm Start) / 序列左移
    // 将时间轴整体左移一格，最后时刻补零或保持最后一个动作
    for (int t = 0; t < config_.horizon - 1; ++t) {
        control_sequence_.col(t) = control_sequence_.col(t + 1);
    }
    control_sequence_.col(config_.horizon - 1).setZero();

    return optimal_control;
}

} // namespace autodrive_garage::mppi