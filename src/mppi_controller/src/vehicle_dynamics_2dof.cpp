#include "mppi_controller/vehicle_dynamics_2dof.hpp"
#include<Eigen/Dense>
#include<cmath>
#include <stdexcept>

namespace autodrive_garage :: dynamics {


std::unique_ptr<BicycleModel2DOF> BicycleModel2DOF::create(const VehicleParams& params) {
    if (params.m <= 0.0 || params.Iz <= 0.0) {
        throw std::invalid_argument("Vehicle mass and inertia must be strictly positive.");
    }
    
    return std::unique_ptr<BicycleModel2DOF>(new BicycleModel2DOF(params));
}

} // namespace autodrive_garage::dynamics 