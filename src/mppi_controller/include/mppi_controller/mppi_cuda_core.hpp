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
                       float x, float y, float yaw, float v, 
                       std::vector<float>& out_costs);

private:
    int num_samples_;
    int horizon_;
    void* d_curand_states_; // 显存指针 (随机数生成器状态)
    float* d_costs_;        // 显存指针 (轨迹代价)
    float* d_ref_traj_;     // 显存指针 (参考路径)
};

} // namespace autodrive_garage::cuda