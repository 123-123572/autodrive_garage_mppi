cmake_minimum_required(VERSION 3.8)
project(mppi_controller)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic -O3 -march=native -ffast-math -fopenmp -DEIGEN_DONT_PARALLELIZE)
endif()

# dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenMP REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(tf2_ros REQUIRED)

add_executable(mppi_control_node
  src/mppi_node.cpp
  src/KinematicBicycleModel.cpp
  src/MPPIController.cpp
  src/vehicle_dynamics_2dof.cpp # <-- 修复点1：让 CMake 和 Clangd 知道这个文件的存在
)

target_include_directories(mppi_control_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${EIGEN3_INCLUDE_DIRS} # <-- 修复点2：确保 Eigen 的路径绝对穿透
)

target_link_libraries(mppi_control_node
  Eigen3::Eigen
  OpenMP::OpenMP_CXX
)

ament_target_dependencies(mppi_control_node
  rclcpp
  nav_msgs
  geometry_msgs
  tf2
  tf2_ros
  tf2_geometry_msgs
)

install(TARGETS
  mppi_control_node
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()