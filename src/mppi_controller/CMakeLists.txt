cmake_minimum_required(VERSION 3.18)
project(mppi_controller LANGUAGES CXX CUDA)

# 设置兼容性更好的显卡架构集合 (75:Turing, 80:Ampere, 86:Ada/Ampere, 89:Ada)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# 显式导出 compile_commands 给 clangd/VSCode
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

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
# 2. 编译底层核心算法库 (mppi_core)
# ==========================================
add_library(mppi_core SHARED
  src/MPPIController.cpp
  src/KinematicBicycleModel.cpp
  src/mppi_cuda_core.cu
  src/hybrid_a_star.cpp
  src/trajectory_processor.cpp
)

target_include_directories(mppi_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${Eigen3_INCLUDE_DIRS}
)

target_link_libraries(mppi_core PUBLIC
  CUDA::cudart
  CUDA::curand
  CUDA::nvToolsExt 
  OpenMP::OpenMP_CXX
  Eigen3::Eigen
)

# ！！！重点修复：这里必须加 PUBLIC ！！！
ament_target_dependencies(mppi_core PUBLIC
  rclcpp nav_msgs geometry_msgs tf2 tf2_ros tf2_geometry_msgs
)

target_compile_options(mppi_core PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -O3 -march=native>
  $<$<COMPILE_LANGUAGE:CUDA>:-O3 -Xcompiler -fPIC>
)

set_target_properties(mppi_core PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_VISIBILITY_PRESET "hidden"
)

# ==========================================
# 3. 编译 ROS 2 节点可执行文件
# ==========================================
macro(mppi_add_node NODE_NAME SOURCE_FILE)
  add_executable(${NODE_NAME} ${SOURCE_FILE})
  
  # ！！！重点修复：这里必须加 PUBLIC ！！！
  target_link_libraries(${NODE_NAME} PUBLIC mppi_core)
  ament_target_dependencies(${NODE_NAME} PUBLIC
    rclcpp nav_msgs geometry_msgs tf2 tf2_ros tf2_geometry_msgs
  )
endmacro()

mppi_add_node(mppi_node src/mppi_node.cpp)
mppi_add_node(hybrid_a_star_node src/hybrid_a_star_node.cpp)

# ==========================================
# 4. ROS 2 安装与导出配置
# ==========================================
ament_export_include_directories(include)
ament_export_libraries(mppi_core)
ament_export_targets(export_mppi_core)
ament_export_dependencies(rclcpp nav_msgs geometry_msgs tf2 tf2_ros tf2_geometry_msgs Eigen3)

install(TARGETS mppi_core
  EXPORT export_mppi_core
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib
)

install(TARGETS mppi_node hybrid_a_star_node
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/
  DESTINATION include
)

ament_package()