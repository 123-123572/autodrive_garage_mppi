#include "mppi_controller/MPPIController.hpp"
#include <cmath>
#include <vector>
#include <omp.h>
#include <iostream>

namespace autodrive_garage::mppi {

MPPIController::ptr MPPIController::create(const Config& config, KinematicBicycleModel::ptr model) {
    return std::make_unique<MPPIController>(config, std::move(model));
}

MPPIController::MPPIController(const Config& config, KinematicBicycleModel::ptr model)
    : config_(config), 
      model_(std::move(model))

{
    // 1. 分配 CPU 侧的内存空间
    control_sequence_ = Eigen::MatrixXd::Zero(2, config_.horizon);
    trajectory_costs_ = Eigen::VectorXd::Zero(config_.num_samples);

    cuda_engine_ = std::make_shared<cuda::CudaMPPIEngine>(config_.num_samples, config_.horizon);
}

ControlVec MPPIController::ComputeControl(const StateVec& current_state, const Eigen::MatrixXd& reference_trajectory) {
    
    // 1. 生成所有样本的控制噪声


    std::vector<float> ref_traj_flat(config_.horizon * 2);
    std::vector<float> ctrl_seq_flat(config_.horizon * 2); 


for (int t = 0; t < config_.horizon; ++t) {
        ref_traj_flat[t * 2]     = static_cast<float>(reference_trajectory(0, t));
        ref_traj_flat[t * 2 + 1] = static_cast<float>(reference_trajectory(1, t));
        
        ctrl_seq_flat[t * 2]     = static_cast<float>(control_sequence_(0, t));
        ctrl_seq_flat[t * 2 + 1] = static_cast<float>(control_sequence_(1, t));
    }

    // 2. 数据展平 (Flatten)：将 Eigen 矩阵转为 C++ 标准的一维 vector，方便传递给 CUDA
    std::vector<float> out_costs(config_.num_samples, 0.0f);
    std::vector<float> out_noise_a(config_.num_samples * config_.horizon, 0.0f);
    std::vector<float> out_noise_s(config_.num_samples * config_.horizon, 0.0f);

    float cur_x = static_cast<float>(current_state(kinematic_bicycle::X));
    float cur_y = static_cast<float>(current_state(kinematic_bicycle::Y));
    float cur_yaw = static_cast<float>(current_state(kinematic_bicycle::YAW));
    float cur_v = static_cast<float>(current_state(kinematic_bicycle::V));

    // ==========================================
    // 🚀 3. 呼叫 GPU 舰队执行极速推演
    // 注意接口变化：我们把基准控制和噪声全送进去了
    // ==========================================
cuda_engine_->launchRollout(ref_traj_flat, ctrl_seq_flat, 
                                cur_x, cur_y, cur_yaw, cur_v, 
                                out_costs, out_noise_a, out_noise_s);

    // 4. 将代价值拷贝回 Eigen 向量
    for(int k = 0; k < config_.num_samples; ++k) {
        trajectory_costs_(k) = static_cast<double>(out_costs[k]);
    }

    // 5. 计算权重 (Information Theoretic Weighting) 
    double min_cost = trajectory_costs_.minCoeff(); 
    double sum_weights = 0.0;
    Eigen::VectorXd weights = Eigen::VectorXd::Zero(config_.num_samples);

    #pragma omp parallel for reduction(+:sum_weights)
    for (int k = 0; k < config_.num_samples; ++k) {
        if (config_.lambda <= 1e-6) throw std::runtime_error("Lambda too small!"); 
        weights(k) = std::exp(-(trajectory_costs_(k) - min_cost) / config_.lambda);
        sum_weights += weights(k);
    }

    // 6. 更新最优控制序列 (使用 GPU 算出来的噪声序列！)
    for (int t = 0; t < config_.horizon; ++t) {
        double delta_accel = 0.0;
        double delta_steer = 0.0;

        #pragma omp parallel for reduction(+:delta_accel, delta_steer)
        for (int k = 0; k < config_.num_samples; ++k) {
            // 注意这里：拿 GPU 返回的一维数组数据
            delta_accel += weights(k) * static_cast<double>(out_noise_a[k * config_.horizon + t]);
            delta_steer += weights(k) * static_cast<double>(out_noise_s[k * config_.horizon + t]);
        }

        control_sequence_(0, t) += delta_accel / sum_weights;
        control_sequence_(1, t) += delta_steer / sum_weights;
    }

    // 7. 提取当前时刻 (t=0) 的控制量输出
    ControlVec optimal_control;
    optimal_control(0) = control_sequence_(0, 0);
    optimal_control(1) = control_sequence_(1, 0);

    // 8. 序列左移 (Warm Start)
    for (int t = 0; t < config_.horizon - 1; ++t) {
        control_sequence_.col(t) = control_sequence_.col(t + 1);
    }
    control_sequence_.col(config_.horizon - 1).setZero();

    return optimal_control;
}

} // namespace autodrive_garage::mppi