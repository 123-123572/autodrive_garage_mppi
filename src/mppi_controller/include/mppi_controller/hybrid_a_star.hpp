#pragma once 


#include <string>
#include <functional>
#include <cstddef>
#include <memory>
#include <vector> 
#include<cmath>
#include<queue>

namespace autodrive_garage::planning {


struct HybridAStarConfig {
    double xy_resolution = 0.2;       // XY 空间网格分辨率 (m)
    double theta_resolution = 0.1;    // 航向角网格分辨率 (rad)
    double step_size = 0.5;           // 每次扩展的步长 (m)
    double wheelbase = 2.8;           // 轴距 (m)
    double max_steer = 0.6;           // 最大前轮转角 (rad)
    int steer_step = 3;               // 采样几个转向角（比如3代表：左、直、右）
    
    // 代价权重
    double forward_penalty = 1.0;     // 前进代价惩罚
    double backward_penalty = 1.5;    // 倒车代价惩罚（通常倒车更难）
    double steer_penalty = 0.5;       // 转向惩罚
    double steer_change_penalty = 1.0;// 转向改变惩罚（防止方向盘乱打）
};

// ==========================================
// 2. 连续-离散混合节点
// ==========================================
struct HybridAStarNode {
    using ptr = std::shared_ptr<HybridAStarNode>;

    // 连续空间状态 (用于真实的物理控制)
    double x = 0.0;
    double y = 0.0;
    double theta = 0.0;

    // 离散空间索引 (用于在 3D Grid 中查重 Closed Set)
    int grid_x = 0;
    int grid_y = 0;
    int grid_theta = 0;

    // A* 代价
    double g_cost = 0.0;  // 从起点到当前点的真实代价
    double h_cost = 0.0;  // 启发式代价 (Heuristic)

    // 控制量 (到达该节点所用的控制)
    double steer = 0.0;
    bool is_forward = true;

    // 父节点指针 (用于回溯路径)
    ptr parent = nullptr;

    // 获取总代价 f = g + h
    [[nodiscard]] double f_cost() const { return g_cost + h_cost; }
};

// ==========================================
// 3. 仿函数：用于优先队列的排序 (小顶堆)
// ==========================================
struct NodeComparator {
    bool operator()(const HybridAStarNode::ptr& lhs, const HybridAStarNode::ptr& rhs) const {
        return lhs->f_cost() > rhs->f_cost(); 
    }
};

// ==========================================
// 4. 仿函数：用于 3D Grid 的哈希计算 (Closed Set)
// ==========================================
struct GridIndexHash {
    size_t operator()(const std::string& key) const {
        return std::hash<std::string>()(key);
    }
};

// ==========================================
// 5. 核心规划器类
// ==========================================
class HybridAStar {
public:
    using ptr = std::unique_ptr<HybridAStar>;

    explicit HybridAStar(const HybridAStarConfig& config);
    ~HybridAStar() = default;

    // 禁用拷贝，保证规划器唯一
    HybridAStar(const HybridAStar&) = delete;
    HybridAStar& operator=(const HybridAStar&) = delete;

    [[nodiscard]] static ptr create(const HybridAStarConfig& config);

    // 主干规划接口
    bool Plan(double start_x, double start_y, double start_theta,
              double goal_x, double goal_y, double goal_theta,
              std::vector<HybridAStarNode::ptr>& out_path);

private:
    HybridAStarConfig config_;

    // 生成唯一哈希键 (字符串拼接法，也可以用位运算提速)
    [[nodiscard]] inline std::string GetIndexKey(int x, int y, int theta) const {
        return std::to_string(x) + "_" + std::to_string(y) + "_" + std::to_string(theta);
    }

    // 坐标连续转离散
    [[nodiscard]] int ComputeGridIndex(double pos, double resolution) const;

    // 运动学扩展函数
    [[nodiscard]] std::vector<HybridAStarNode::ptr> ExpandNode(const HybridAStarNode::ptr& current_node);

    // 碰撞检测 (桩函数，需接入 Costmap)
    [[nodiscard]] bool IsCollisionFree(double x, double y, double theta) const;

    // 启发式函数计算 (无障碍 Reed-Shepp 距离 + 2D A* 距离)
    [[nodiscard]] double CalculateHeuristic(const HybridAStarNode::ptr& node, 
                                            double goal_x, double goal_y, double goal_theta) const;
};

} // namespace autodrive_garage::planning

