#ifndef AUTODRIVE_GARAGE_BICYCLE_MODEL_2DOF_HPP
#define AUTODRIVE_GARAGE_BICYCLE_MODEL_2DOF_HPP

#include <memory>
#include <cmath>

namespace autodrive_garage :: dynamics {

// 车辆物理参数 (使用常量表达式确保可以在编译期优化)
struct VehicleParams {
    double m;   // 车辆质量 [kg]
    double Iz;  // 横摆转动惯量 [kg*m^2]
    double lf;  // 质心到前轴距离 [m]
    double lr;  // 质心到后轴距离 [m]
    double Cf;  // 前轴侧偏刚度 [N/rad] 
    double Cr;  // 后轴侧偏刚度 [N/rad]
};

// 车辆运动学与动力学状态
struct VehicleState {
    double e_y;           // 横向误差 [m]
    double e_y_dot;       // 横向误差率 [m/s]
    double e_theta;       // 航向误差 [rad]
    double e_theta_dot;   // 航向误差率 [rad/s]
    double vx;            // 当前纵向速度 [m/s] 
    double kappa;         // 参考路径点曲率 [1/m] (用于计算 theta_ref_dot)
};
class BicycleModel2DOF {
public:
    [[nodiscard]] static std::unique_ptr<BicycleModel2DOF> create(const VehicleParams& params);

    // 禁用拷贝构造和赋值操作，保证实例的唯一性，提升安全性
    BicycleModel2DOF(const BicycleModel2DOF&) = delete;
    BicycleModel2DOF& operator=(const BicycleModel2DOF&) = delete;

    // 默认移动语义
    BicycleModel2DOF(BicycleModel2DOF&&) = default;
    BicycleModel2DOF& operator=(BicycleModel2DOF&&) = default;
    ~BicycleModel2DOF() = default;

    /**
     * @brief 更新车辆状态 (核心计算函数)
     * @param state 当前状态 (将会被就地修改，避免拷贝开销)
     * @param steer_angle_f 前轮转角 [rad]
     * @param dt 采样时间 [s]
     * @note 声明为 inline 和 noexcept 帮助编译器进行极致的内联展开和优化
     */
    inline void updateState(VehicleState& state, double steer_angle_f, double dt) const noexcept;

private:
    explicit BicycleModel2DOF(const VehicleParams& params) : params_(params) {}
    VehicleParams params_;
};

//内联函数展开
inline void BicycleModel2DOF::updateState(VehicleState& state, double steer_angle_f, double dt) const noexcept {
    // 防止vx降成0
    const double vx = std::max(state.vx, 0.1); 
    const double inv_vx = 1.0 / vx;
    
    // 参考路径航向角速度
    const double theta_ref_dot = vx * state.kappa;


    
    const double e_y_ddot = 
        - ((params_.Cf + params_.Cr) / params_.m * inv_vx) * state.e_y_dot 
        + ((params_.Cf + params_.Cr) / params_.m) * state.e_theta 
        - ((params_.Cf * params_.lf - params_.Cr * params_.lr) / params_.m * inv_vx) * state.e_theta_dot 
        + (params_.Cf / params_.m) * steer_angle_f 
        - ((params_.Cf * params_.lf - params_.Cr * params_.lr) / params_.m * inv_vx + vx) * theta_ref_dot;

    const double e_theta_ddot = 
        - ((params_.Cf * params_.lf - params_.Cr * params_.lr) / params_.Iz * inv_vx) * state.e_y_dot 
        + ((params_.Cf * params_.lf - params_.Cr * params_.lr) / params_.Iz) * state.e_theta 
        - ((params_.Cf * params_.lf * params_.lf + params_.Cr * params_.lr * params_.lr) / params_.Iz * inv_vx) * state.e_theta_dot 
        + ((params_.Cf * params_.lf) / params_.Iz) * steer_angle_f 
        - ((params_.Cf * params_.lf * params_.lf + params_.Cr * params_.lr * params_.lr) / params_.Iz * inv_vx) * theta_ref_dot;

    // 4. 前向欧拉积分更新误差状态
    state.e_y += state.e_y_dot * dt;
    state.e_y_dot += e_y_ddot * dt;
    
    state.e_theta += state.e_theta_dot * dt;
    state.e_theta_dot += e_theta_ddot * dt;
}


} // namespace dynamics :: autodrive_garage

#endif // AUTODRIVE_GARAGE_BICYCLE_MODEL_2DOF_HPP