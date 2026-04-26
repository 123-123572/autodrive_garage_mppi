#include "mppi_controller/MPPIController.hpp"
#include <cmath>
#include <vector>
#include <algorithm>
#include <iostream>

namespace autodrive_garage::mppi {

MPPIController::ptr MPPIController::create(const Config& config, KinematicBicycleModel::ptr model) {
    return std::make_unique<MPPIController>(config, std::move(model));
}

MPPIController::MPPIController(const Config& config, KinematicBicycleModel::ptr model)
    : config_(config), 
      model_(std::move(model)) 
{
    // 1. 初始化 Eigen 容器
    control_sequence_ = Eigen::MatrixXd::Zero(2, config_.horizon);
    trajectory_costs_ = Eigen::VectorXd::Zero(config_.num_samples);

    // 2. 为成员变量缓冲区预分配固定空间，消除运行时的内存抖动
    h_ref_traj_.resize(config_.horizon * 2);
    h_base_ctrl_.resize(config_.horizon * 2);
    h_costs_.resize(config_.num_samples);
    h_noise_a_.resize(config_.num_samples * config_.horizon);
    h_noise_s_.resize(config_.num_samples * config_.horizon);

    // 3. 启动 CUDA 引擎
    cuda_engine_ = std::make_shared<cuda::CudaMPPIEngine>(config_.num_samples, config_.horizon);
}

ControlVec MPPIController::ComputeControl(const StateVec& current_state, const Eigen::MatrixXd& reference_trajectory) {
    
    // 🚀 [性能点 1]：不再创建临时 std::vector，直接写入预分配好的成员变量 h_xxx
    for (int t = 0; t < config_.horizon; ++t) {
        h_ref_traj_[t * 2]     = static_cast<float>(reference_trajectory(0, t));
        h_ref_traj_[t * 2 + 1] = static_cast<float>(reference_trajectory(1, t));
        
        h_base_ctrl_[t * 2]     = static_cast<float>(control_sequence_(0, t));
        h_base_ctrl_[t * 2 + 1] = static_cast<float>(control_sequence_(1, t));
    }

    // 🚀 [性能点 2]：调用 GPU 推演，直接把成员变量的地址喂给接口
    cuda_engine_->launchRollout(
        h_ref_traj_, h_base_ctrl_, 
        static_cast<float>(current_state(kinematic_bicycle::X)), 
        static_cast<float>(current_state(kinematic_bicycle::Y)), 
        static_cast<float>(current_state(kinematic_bicycle::YAW)), 
        static_cast<float>(current_state(kinematic_bicycle::V)), 
        h_costs_, h_noise_a_, h_noise_s_);

    // 🚀 [性能点 3]：利用 Eigen::Map 实现“零拷贝”映射，将 1D 数组视为矩阵操作
    Eigen::Map<Eigen::VectorXf> costs_map(h_costs_.data(), config_.num_samples);
    
    // 计算权重：w = exp(-(cost - min) / lambda)
    float min_cost = costs_map.minCoeff();
    Eigen::VectorXf weights = (-(costs_map.array() - min_cost) / static_cast<float>(config_.lambda)).exp();
    float sum_weights = weights.sum();

    if (sum_weights < 1e-6f) return control_sequence_.col(0); // 防爆保护
    weights /= sum_weights; // 归一化

    // 🚀 [性能点 4]：降维打击！用矩阵乘法代替嵌套 OpenMP 循环
    // 将一维噪声数组映射为 [num_samples x horizon] 的矩阵（RowMajor 是关键，因为 CUDA 数据是行主序存储的）
    Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> 
        noise_a_mat(h_noise_a_.data(), config_.num_samples, config_.horizon);
    Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> 
        noise_s_mat(h_noise_s_.data(), config_.num_samples, config_.horizon);

    // 加权求和公式：最优噪声序列 = 权重向量 * 噪声矩阵
    // 这行代码会自动调用 CPU 的 SIMD (AVX/SSE) 指令集，比手动写 loop 快得多！
    Eigen::VectorXf delta_accel_seq = weights.transpose() * noise_a_mat;
    Eigen::VectorXf delta_steer_seq = weights.transpose() * noise_s_mat;

    // 更新基准控制序列
    for (int t = 0; t < config_.horizon; ++t) {
        control_sequence_(0, t) += static_cast<double>(delta_accel_seq(t));
        control_sequence_(1, t) += static_cast<double>(delta_steer_seq(t));
    }

    // 7. 提取 t=0 的控制量
    ControlVec optimal_control = control_sequence_.col(0);

    // 8. 序列左移 (Warm Start) - 使用 Eigen 的 block 操作实现快速移动
    int h = config_.horizon;
    control_sequence_.block(0, 0, 2, h - 1) = control_sequence_.block(0, 1, 2, h - 1);
    control_sequence_.col(h - 1).setZero();

    return optimal_control;
}

} // namespace autodrive_garage::mppi