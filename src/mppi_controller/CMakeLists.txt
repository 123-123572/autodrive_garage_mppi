cmake_minimum_required(VERSION 3.18)
project(mppi_controller LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# 显式导出 compile_commands 给 clangd
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# 视显卡修改，如果支持更高版本 CUDA，可以设置更高的版本
set(CMAKE_CUDA_ARCHITECTURES 75)

# 查找所需的依赖项
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(OpenMP REQUIRED)

# 添加核心库
add_library(mppi_core SHARED
  src/MPPIController.cpp
  src/KinematicBicycleModel.cpp
  src/mppi_cuda_core.cu
)

# 设置头文件路径
target_include_directories(mppi_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

# 链接所需库
target_link_libraries(mppi_core PUBLIC
  CUDA::cudart
  CUDA::curand
  OpenMP::OpenMP_CXX
  Eigen3::Eigen
)

# 编译选项，启用 CUDA 优化
target_compile_options(mppi_core PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic -O3>
  $<$<COMPILE_LANGUAGE:CUDA>:-O3>
)

# 启用 CUDA 可分离编译
set_target_properties(mppi_core PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
)

# 设置节点目标
add_executable(mppi_node src/mppi_node.cpp)

# 设置节点的头文件路径
target_include_directories(mppi_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

# 链接核心库
target_link_libraries(mppi_node
  mppi_core
)

# 依赖项配置
ament_target_dependencies(mppi_node
  rclcpp
  nav_msgs
  geometry_msgs
  tf2
  tf2_geometry_msgs
)

# 安装目标
install(TARGETS mppi_core mppi_node
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

# 安装头文件
install(DIRECTORY include/
  DESTINATION include
)

ament_package()