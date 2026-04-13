#pragma once

#include <Eigen/Dense>
#include <memory>
#include <cmath>


namespace autodrive_garage :: kinematic_bicycle {

// 状态量枚举
enum StateIndex { X = 0, Y = 1, YAW = 2, V = 3 };
// 控制量枚举
enum ControlIndex { ACCEL = 0, STEER = 1 };

// 定义定长 Eigen
using StateVec = Eigen::Vector4d;
using ControlVec = Eigen::Vector2d;

class KinematicBicycleModel {
public:
    using ptr = std::unique_ptr<KinematicBicycleModel>;

    struct Config {
        double wheelbase = 2.8;      // 轴距 (m)
        double max_steer = 0.6;      // 最大前轮转角 (rad)
        double min_steer = -0.6;     // 最小前轮转角 (rad)
        double max_accel = 3.0;      // 最大加速度 (m/s^2)
        double min_accel = -5.0;     // 最大减速度 (m/s^2)

    };

    explicit KinematicBicycleModel(const Config& config);
    ~KinematicBicycleModel() = default;

    KinematicBicycleModel(const KinematicBicycleModel&) = delete;
    KinematicBicycleModel& operator=(const KinematicBicycleModel&) = delete;
    
    [[nodiscard]] static ptr create(const Config& config);

    //更新状态函数
    [[nodiscard]] StateVec UpdateState(const StateVec& current_state, 
                                       const ControlVec& control, 
                                       double dt) const;

    //获取车辆配置
    [[nodiscard]] const Config& GetConfig() const noexcept { return config_; }
private:
    Config config_;

    [[nodiscard]] inline double Clamp(double value, double min_val, double max_val) const {
        return std::max(min_val, std::min(value, max_val));
    }
};
}