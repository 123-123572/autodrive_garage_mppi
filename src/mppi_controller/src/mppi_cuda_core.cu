#include <vector>
#include <memory>
#include <iostream>

#include <cuda_runtime.h>
#include <curand_kernel.h>


#include "mppi_controller/mppi_cuda_core.hpp"


namespace autodrive_garage::cuda {

// 车辆参数常驻常量内存 
struct DeviceConfig {
    float wheelbase = 2.8f;
    float dt = 0.02f;
    int horizon = 50;
    float ref_v = 5.0f; // 目标速度
};

__constant__ DeviceConfig d_config;

 //cuRAND 随机数状态初始化 Kernel
__global__ void InitCurandKernel(curandState* state, unsigned long seed, int num_samples) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_samples) {
        curand_init(seed, idx, 0, &state[idx]);
    }
}

//MPPI核心推演 Kernel 
__global__ void MPPIRolloutKernel(
    curandState* curand_states,
    float* d_costs,             // 输出：每条轨迹的代价 [num_samples]
    const float* d_ref_traj,    // 输入：参考路径 [horizon * 2] (x, y交替)
    float cur_x, float cur_y, float cur_yaw, float cur_v,
    float std_dev_accel, float std_dev_steer,
    int num_samples) 
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= num_samples) return;

    // 线程私有寄存器：当前轨迹的状态拷贝
    float x = cur_x;
    float y = cur_y;
    float yaw = cur_yaw;
    float v = cur_v;
    float cost = 0.0f;

    // 获取当前线程专属的随机数生成器状态
    curandState local_state = curand_states[k];

    for (int t = 0; t < d_config.horizon; ++t) {
        // 生成纯正的 GPU 高斯噪声
        float a_noise = curand_normal(&local_state) * std_dev_accel;
        float s_noise = curand_normal(&local_state) * std_dev_steer;

        // 模型更新
        x += v * cosf(yaw) * d_config.dt;
        y += v * sinf(yaw) * d_config.dt;
        yaw += (v / d_config.wheelbase) * tanf(s_noise) * d_config.dt;
        v += a_noise * d_config.dt;

        // 轨迹代价计算
        float ref_x = d_ref_traj[t * 2];
        float ref_y = d_ref_traj[t * 2 + 1];
        float dx = x - ref_x;
        float dy = y - ref_y;
        
        cost += (dx * dx + dy * dy); 
    }

    // 保存该线程专属的最终代价，并将随机数状态写回显存
    d_costs[k] = cost;
    curand_states[k] = local_state; 
}

CudaMPPIEngine::CudaMPPIEngine(int samples, int horizon) 
    : num_samples_(samples), horizon_(horizon) { 
    
    DeviceConfig cfg;
    cfg.horizon = horizon;
    cudaMemcpyToSymbol(d_config, &cfg, sizeof(DeviceConfig));

    cudaMalloc(&d_curand_states_, num_samples_ * sizeof(curandState));//随机数状态
    cudaMalloc(&d_costs_, num_samples_ * sizeof(float));//代价函数
    cudaMalloc(&d_ref_traj_, horizon_ * 2 * sizeof(float));//参考路径

    int tpb = 256;
    int blocks = (num_samples_ + tpb - 1) / tpb;
    InitCurandKernel<<<blocks, tpb>>>((curandState*)d_curand_states_, 1234ULL, num_samples_);
}

CudaMPPIEngine::~CudaMPPIEngine() {
    if (d_curand_states_) cudaFree(d_curand_states_);
    if (d_costs_) cudaFree(d_costs_);
    if (d_ref_traj_) cudaFree(d_ref_traj_);
}

void CudaMPPIEngine::launchRollout(const std::vector<float>& ref_traj, 
                                  float x, float y, float yaw, float v, 
                                  std::vector<float>& out_costs) {
    cudaMemcpy(d_ref_traj_, ref_traj.data(), horizon_ * 2 * sizeof(float), cudaMemcpyHostToDevice);

    int tpb = 256;
    int blocks = (num_samples_ + tpb - 1) / tpb;
    MPPIRolloutKernel<<<blocks, tpb>>>(
        (curandState*)d_curand_states_, d_costs_, d_ref_traj_, 
        x, y, yaw, v, 0.5f, 0.1f, num_samples_
    );
    
    cudaMemcpy(out_costs.data(), d_costs_, num_samples_ * sizeof(float), cudaMemcpyDeviceToHost);
}

} // namespace autodrive_garage::cuda