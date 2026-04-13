#include "mppi_controller/KinematicBicycleModel.hpp"

namespace autodrive_garage::kinematic_bicycle {

KinematicBicycleModel::KinematicBicycleModel(const Config& config)
    : config_(config) {}

KinematicBicycleModel::ptr KinematicBicycleModel::create(const Config& config) {
    return std::make_unique<KinematicBicycleModel>(config);
}

StateVec KinematicBicycleModel::UpdateState(const StateVec& current_state,
                                            const ControlVec& control,
                                            double dt) const {
    StateVec next_state;

    // 限制输入控制
    double accel = Clamp(control(ACCEL), config_.min_accel, config_.max_accel);
    double steer = Clamp(control(STEER), config_.min_steer, config_.max_steer);

    // 提取当前状态
    double x   = current_state(X);
    double y   = current_state(Y);
    double yaw = current_state(YAW);
    double v   = current_state(V);

    // 求导
    double dx   = v * std::cos(yaw);
    double dy   = v * std::sin(yaw);
    double dyaw = (v / config_.wheelbase) * std::tan(steer);
    double dv   = accel;

    // 欧拉积分更新状态
    next_state(X) = x + dx * dt;
    next_state(Y) = y + dy * dt;

    double next_yaw = yaw + dyaw * dt;
    next_yaw = std::fmod(next_yaw + M_PI, 2.0 * M_PI);
    if (next_yaw < 0) {
        next_yaw += 2.0 * M_PI;
    }
    next_state(YAW) = next_yaw - M_PI;

    next_state(V) = v + dv * dt;

    return next_state;
}

}  // namespace autodrive_garage::kinematic_bicycle