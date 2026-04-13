#pragma once

#include <Eigen/Dense>
#include <memory>
#include <random>
#include "mppi_controller/KinematicBicycleModel.hpp"


namespace autodrive_garage::mppi {

using kinematic_bicycle::KinematicBicycleModel;
using kinematic_bicycle::StateVec;
using kinematic_bicycle::ControlVec;

class MPPIController {
public:
    using ptr = std::unique_ptr<MPPIController>;

    struct Config {
        int num_samples = 1000;       // K: 采样轨迹数量
        int horizon = 50;             // T: 预测时域步数
        double dt = 0.1;              // 离散时间间隔
        double lambda = 1.0;          // 温度参数
        
        // 噪声标准差 (Sigma)
        double std_dev_accel = 1.0; 
        double std_dev_steer = 0.2;
    };

    [[nodiscard]] static ptr create(const Config& config, KinematicBicycleModel::ptr model);

    explicit MPPIController(const Config& config, KinematicBicycleModel::ptr model);
    ~MPPIController() = default;

    MPPIController(const MPPIController&) = delete;
    MPPIController& operator=(const MPPIController&) = delete;

    // 核心控制接口：输入当前状态和参考轨迹，输出下一步的最优控制量
    [[nodiscard]] ControlVec ComputeControl(const StateVec& current_state, const Eigen::MatrixXd& reference_trajectory);

private:
    Config config_;
    KinematicBicycleModel::ptr model_;

    
    Eigen::MatrixXd control_sequence_;       // [2 x horizon]，保存上一帧的控制序列 u
    Eigen::MatrixXd noise_accel_;            // [num_samples x horizon]，加速度噪声矩阵
    Eigen::MatrixXd noise_steer_;            // [num_samples x horizon]，转向噪声矩阵
    Eigen::VectorXd trajectory_costs_;       // [num_samples]，存储每条轨迹的代价 S_k

    // 随机数生成器
    std::mt19937 rng_;
    std::normal_distribution<double> dist_accel_;
    std::normal_distribution<double> dist_steer_;

    // 内部私有方法
    void GenerateNoise();
    double ComputeTrajectoryCost(const StateVec& final_state, const Eigen::MatrixXd& reference_trajectory, int t);
};

} // namespace autodrive_garage::mppi