#include <vector>
#include <memory>
#include <iostream>

#include <cuda_runtime.h>
#include <curand_kernel.h>


#include "mppi_controller/mppi_cuda_core.hpp"


namespace autodrive_garage::cuda {

// 车辆参数常驻常量内存 
// 在 mppi_cuda_core.cu 中更新
struct DeviceConfig {
    float wheelbase = 2.8f;
    float dt = 0.02f;
    int horizon = 50;
    float ref_v = 5.0f; 

    // === 新增：代价函数权重 ===
    float w_x = 10.0f;       // 横向跟踪
    float w_y = 10.0f;       // 纵向跟踪
    float w_yaw = 5.0f;      // 航向角误差
    float w_v = 2.0f;        // 速度保持
    float w_steer = 100.0f;  // 惩罚大方向盘
    float w_accel = 10.0f;   // 惩罚大油门
    float w_jerk = 50.0f;    // 惩罚加速度突变 (平顺性核心)
    float w_steer_rate = 200.0f; // 惩罚方向盘狂打 (平顺性核心)
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
// 替换原有的 MPPIRolloutKernel
__global__ void MPPIRolloutKernel(
    curandState* curand_states,
   float* d_costs,             
    const float* d_ref_traj,    
    const float* d_base_ctrl,   // [新增] 输入：上一帧的基准控制序列
    float* d_out_noise_a,       // [新增] 输出：GPU生成的加速度噪声
    float* d_out_noise_s,       // [新增] 输出：GPU生成的转向噪声
    float cur_x, float cur_y, float cur_yaw, float cur_v,
    float std_dev_accel, float std_dev_steer,
    int num_samples)
{
// 1. 声明动态共享内存 (存参考轨迹 X 和 Y)
    extern __shared__ float s_ref_traj[]; 

    int tid = threadIdx.x;
    int k = blockIdx.x * blockDim.x + tid;

    int total_traj_elements = d_config.horizon * 2;
    if (tid < total_traj_elements) {
        s_ref_traj[tid] = d_ref_traj[tid];
    }
    // 关键！必须等所有线程搬运完毕才能往下走
    __syncthreads();

    if (k >= num_samples) return;

    // 线程私有状态
    float x = cur_x;
    float y = cur_y;
    float yaw = cur_yaw;
    float v = cur_v;
    float cost = 0.0f;
    // 用于计算控制平顺性 (Jerk 和 Steer Rate)
    float prev_a = 0.0f; 
    float prev_s = 0.0f;

    curandState local_state = curand_states[k];

    // 3. 开始前向推演
    for (int t = 0; t < d_config.horizon; ++t) {
        // 生成控制指令 (目前是纯噪声，后续可加上 base_control)
        // [修改核心逻辑]：生成纯噪声，并保存到显存以便传回 CPU
        float n_a = curand_normal(&local_state) * std_dev_accel;
        float n_s = curand_normal(&local_state) * std_dev_steer;
        
        d_out_noise_a[k * d_config.horizon + t] = n_a;
        d_out_noise_s[k * d_config.horizon + t] = n_s;

        // 实际推演控制量 = 基准控制量 + 噪声
        float cur_a = d_base_ctrl[t * 2] + n_a;
        float cur_s = d_base_ctrl[t * 2 + 1] + n_s;

        // --- 物理推演 ---
        x += v * cosf(yaw) * d_config.dt;
        y += v * sinf(yaw) * d_config.dt;
        yaw += (v / d_config.wheelbase) * tanf(cur_s) * d_config.dt;
        v += cur_a * d_config.dt;

        // 规范化航向角到 [-pi, pi] (根据需要添加)

        // --- 代价计算开始 ---
        
        // A. 状态跟踪代价 (从超快 Shared Memory 读取!)
        float ref_x = s_ref_traj[t * 2];
        float ref_y = s_ref_traj[t * 2 + 1];
        
        // 简单假设参考航向角为路径点连线的切向 (这里简化，实际最好从 CPU 传进来)
        float ref_yaw = atan2f(ref_y - y, ref_x - x); 
        
        float dx = x - ref_x;
        float dy = y - ref_y;
        float dyaw = yaw - ref_yaw;
        float dv = v - d_config.ref_v;

        float stage_cost = d_config.w_x * (dx * dx) + 
                           d_config.w_y * (dy * dy) + 
                           d_config.w_yaw * (dyaw * dyaw) + 
                           d_config.w_v * (dv * dv);

        // B. 平顺性代价 (控制量本身 + 变化率)
        float delta_a = cur_a - prev_a;
        float delta_s = cur_s - prev_s;

        stage_cost += d_config.w_accel * (cur_a * cur_a) + 
                      d_config.w_steer * (cur_s * cur_s) + 
                      d_config.w_jerk * (delta_a * delta_a) + 
                      d_config.w_steer_rate * (delta_s * delta_s);

        // C. 终端代价提升 (Myopia 修正)
        if (t == d_config.horizon - 1) {
            stage_cost *= 5.0f; // 终端步权重放大 5 倍
        }

        cost += stage_cost;

        // 更新历史控制量
        prev_a = cur_a;
        prev_s = cur_s;
    }

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
    // [新增] 分配基准控制和噪声输出的显存
    cudaMalloc(&d_base_ctrl_, horizon_ * 2 * sizeof(float));
    cudaMalloc(&d_noise_a_, num_samples_ * horizon_ * sizeof(float));
    cudaMalloc(&d_noise_s_, num_samples_ * horizon_ * sizeof(float));
    int tpb = 256;
    int blocks = (num_samples_ + tpb - 1) / tpb;
    InitCurandKernel<<<blocks, tpb>>>((curandState*)d_curand_states_, 1234ULL, num_samples_);
}

CudaMPPIEngine::~CudaMPPIEngine() {
if (d_curand_states_) cudaFree(d_curand_states_);
    if (d_costs_) cudaFree(d_costs_);
    if (d_ref_traj_) cudaFree(d_ref_traj_);
    if (d_base_ctrl_) cudaFree(d_base_ctrl_);
    if (d_noise_a_) cudaFree(d_noise_a_);
    if (d_noise_s_) cudaFree(d_noise_s_);
}

void CudaMPPIEngine::launchRollout(const std::vector<float>& ref_traj, 
                                   const std::vector<float>& base_ctrl,
                                   float x, float y, float yaw, float v, 
                                   std::vector<float>& out_costs,
                                   std::vector<float>& out_noise_a,
                                   std::vector<float>& out_noise_s) {
    cudaMemcpy(d_ref_traj_, ref_traj.data(), horizon_ * 2 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_base_ctrl_, base_ctrl.data(), horizon_ * 2 * sizeof(float), cudaMemcpyHostToDevice);

    int tpb = 256;
    int blocks = (num_samples_ + tpb - 1) / tpb;
    
    // 计算 Shared Memory 的字节数: horizon * 2 个 float
    size_t shared_mem_bytes = horizon_ * 2 * sizeof(float);

    // 挂载第三个参数！
MPPIRolloutKernel<<<blocks, tpb, shared_mem_bytes>>>(
        (curandState*)d_curand_states_, d_costs_, d_ref_traj_, d_base_ctrl_, 
        d_noise_a_, d_noise_s_,
        x, y, yaw, v, 0.5f, 0.1f, num_samples_
    );
    
    cudaMemcpy(out_costs.data(), d_costs_, num_samples_ * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(out_noise_a.data(), d_noise_a_, num_samples_ * horizon_ * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(out_noise_s.data(), d_noise_s_, num_samples_ * horizon_ * sizeof(float), cudaMemcpyDeviceToHost);
}

} // namespace autodrive_garage::cuda