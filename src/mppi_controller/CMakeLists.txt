cmake_minimum_required(VERSION 3.18)

project(mppi_controller LANGUAGES CXX CUDA)

# ==========================================
# 0. 基础编译配置
# ==========================================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89 120)
endif()

# ==========================================
# 1. 查找依赖项
# ==========================================
find_package(ament_cmake REQUIRED)

find_package(rclcpp REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(OpenMP REQUIRED)

# ==========================================
# 2. 编译核心库 mppi_core
# ==========================================
add_library(mppi_core SHARED
  src/MPPIController.cpp
  src/KinematicBicycleModel.cpp
  src/mppi_cuda_core.cu
  src/hybrid_a_star.cpp
  src/trajectory_processor.cpp
  third_party/dubins/dubins.cpp
)

target_include_directories(mppi_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/third_party/dubins>
)
ament_target_dependencies(mppi_core
  rclcpp
  nav_msgs
  geometry_msgs
  tf2
  tf2_ros
  tf2_geometry_msgs
)

target_link_libraries(mppi_core
  Eigen3::Eigen
  CUDA::cudart
  CUDA::curand
  CUDA::nvToolsExt
  OpenMP::OpenMP_CXX
)

target_compile_options(mppi_core PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -O3 -march=native>
  $<$<COMPILE_LANGUAGE:CUDA>:-O3 -Xcompiler=-fPIC>
)

set_target_properties(mppi_core PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# ==========================================
# 3. 编译 ROS2 节点
# ==========================================
function(mppi_add_node NODE_NAME SOURCE_FILE)
  add_executable(${NODE_NAME} ${SOURCE_FILE})

  ament_target_dependencies(${NODE_NAME}
    rclcpp
    nav_msgs
    geometry_msgs
    tf2
    tf2_ros
    tf2_geometry_msgs
  )

  target_link_libraries(${NODE_NAME} mppi_core)

  target_compile_options(${NODE_NAME} PRIVATE
    -Wall
    -Wextra
    -O3
    -march=native
  )
endfunction()

mppi_add_node(mppi_node src/mppi_node.cpp)
mppi_add_node(hybrid_a_star_node src/hybrid_a_star_node.cpp)

# ==========================================
# 4. 安装
# ==========================================
install(TARGETS mppi_core
  EXPORT export_mppi_core
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(TARGETS
  mppi_node
  hybrid_a_star_node
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/
  DESTINATION include
)

# ==========================================
# 5. 导出
# ==========================================
ament_export_include_directories(include)
ament_export_libraries(mppi_core)
ament_export_targets(export_mppi_core HAS_LIBRARY_TARGET)

ament_export_dependencies(
  rclcpp
  nav_msgs
  geometry_msgs
  tf2
  tf2_ros
  tf2_geometry_msgs
  Eigen3
)

ament_package()