#pragma once
#include <vector>
#include <memory>

namespace autodrive_garage::cuda {

class CudaMPPIEngine {
public:
    using ptr = std::shared_ptr<CudaMPPIEngine>;
    
    CudaMPPIEngine(int samples, int horizon);
    ~CudaMPPIEngine();

    // 接口
    void launchRollout(const std::vector<float>& ref_traj, 
                                   const std::vector<float>& base_ctrl,
                                   float x, float y, float yaw, float v, 
                                   std::vector<float>& out_costs,
                                   std::vector<float>& out_noise_a,
                                   std::vector<float>& out_noise_s);

private:
    int num_samples_;
    int horizon_;

    void* d_curand_states_ = nullptr; // 显存指针 (随机数生成器状态)
    float* d_costs_ = nullptr;        // 显存指针 (轨迹代价)
    float* d_ref_traj_ = nullptr;     // 显存指针 (参考路径)
    float* d_base_ctrl_ = nullptr;    // 显存指针 (基准控制序列)
    float* d_noise_a_ = nullptr;      // 显存指针 (加速度噪声)
    float* d_noise_s_ = nullptr;      // 显存指针 (转向噪声)
};

} // namespace autodrive_garage::cuda