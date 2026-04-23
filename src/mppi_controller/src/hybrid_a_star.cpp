#include "mppi_controller/hybrid_a_star.hpp"
#include <iostream>
#include <algorithm>

namespace autodrive_garage::planning {

HybridAStar::HybridAStar(const HybridAStarConfig& config) : config_(config) {}

HybridAStar::ptr HybridAStar::create(const HybridAStarConfig& config) {
    return std::make_unique<HybridAStar>(config);
}

int HybridAStar::ComputeGridIndex(double pos, double resolution) const {
    return static_cast<int>(std::floor(pos / resolution));
}

bool HybridAStar::Plan(double start_x, double start_y, double start_theta,
                       double goal_x, double goal_y, double goal_theta,
                       std::vector<HybridAStarNode::ptr>& out_path) {
    
    // 1. 初始化 Open Set (优先队列) 和 Closed Set (哈希表)
    std::priority_queue<HybridAStarNode::ptr, 
                        std::vector<HybridAStarNode::ptr>, 
                        NodeComparator> open_set;
    
    std::unordered_map<std::string, HybridAStarNode::ptr, GridIndexHash> closed_set;

    // 2. 创建起点 Node
    auto start_node = std::make_shared<HybridAStarNode>();
    start_node->x = start_x;
    start_node->y = start_y;
    start_node->theta = start_theta;
    start_node->grid_x = ComputeGridIndex(start_x, config_.xy_resolution);
    start_node->grid_y = ComputeGridIndex(start_y, config_.xy_resolution);
    start_node->grid_theta = ComputeGridIndex(start_theta, config_.theta_resolution);
    start_node->g_cost = 0.0;
    start_node->h_cost = CalculateHeuristic(start_node, goal_x, goal_y, goal_theta);
    start_node->parent = nullptr;

    open_set.push(start_node);
    
    // 记录到 Closed Set
    closed_set[GetIndexKey(start_node->grid_x, start_node->grid_y, start_node->grid_theta)] = start_node;

    int iter_count = 0;
    const int MAX_ITERATIONS = 100000; // 防死循环

    // 3. A* 主循环
    while (!open_set.empty() && iter_count < MAX_ITERATIONS) {
        iter_count++;

        // 弹出 f_cost 最小的节点
        auto current = open_set.top();
        open_set.pop();

        // 到达目标点判断 (容差可以自行配置)
        if (std::hypot(current->x - goal_x, current->y - goal_y) < config_.xy_resolution) {
            std::cout << "🚀 Hybrid A* 找到终点！迭代次数: " << iter_count << std::endl;
            
            // 回溯路径
            auto ptr = current;
            while (ptr != nullptr) {
                out_path.push_back(ptr);
                ptr = ptr->parent;
            }
            std::reverse(out_path.begin(), out_path.end());
            return true;
        }

        // 4. 节点扩展 (前向/后向推演)
        auto next_nodes = ExpandNode(current);

        for (auto& next : next_nodes) {
            // 碰撞检测
            if (!IsCollisionFree(next->x, next->y, next->theta)) {
                continue; 
            }

            // 计算该离散网格的唯一 Key
            std::string key = GetIndexKey(next->grid_x, next->grid_y, next->grid_theta);

            // 如果该网格没被访问过，或者找到了更优的 g_cost 路径
            if (closed_set.find(key) == closed_set.end() || next->g_cost < closed_set[key]->g_cost) {
                
                next->h_cost = CalculateHeuristic(next, goal_x, goal_y, goal_theta);
                
                closed_set[key] = next; // 霸占或更新这个网格
                open_set.push(next);    // 扔进候选池
            }
        }
    }

    std::cerr << "❌ Hybrid A* 搜索失败或达到最大迭代次数。" << std::endl;
    return false;
}

std::vector<HybridAStarNode::ptr> HybridAStar::ExpandNode(const HybridAStarNode::ptr& current) {
    std::vector<HybridAStarNode::ptr> expanded_nodes;
    
    // 转向角等分采样 (例如: -0.6, 0.0, 0.6)
    double steer_step_val = (2.0 * config_.max_steer) / (config_.steer_step - 1);

    // 两个方向：1 代表前进，-1 代表倒车
    std::vector<int> directions = {1, -1};

    for (int dir : directions) {
        for (int i = 0; i < config_.steer_step; ++i) {
            double steer = -config_.max_steer + i * steer_step_val;

            auto next = std::make_shared<HybridAStarNode>();
            
            // ----------------------------------------------------
            // 物理引擎：离散化的运动学推演 (Bicycle Model)
            // ----------------------------------------------------
            double travel_dist = dir * config_.step_size;
            
            next->x = current->x + travel_dist * std::cos(current->theta);
            next->y = current->y + travel_dist * std::sin(current->theta);
            
            // 航向角更新： d_theta = (v / L) * tan(steer) * dt 
            // 这里我们用步长代替 v*dt： d_theta = (travel_dist / L) * tan(steer)
            double d_theta = (travel_dist / config_.wheelbase) * std::tan(steer);
            next->theta = current->theta + d_theta;

            // 角度归一化到 [-pi, pi]
            next->theta = std::atan2(std::sin(next->theta), std::cos(next->theta));

            // 更新离散索引
            next->grid_x = ComputeGridIndex(next->x, config_.xy_resolution);
            next->grid_y = ComputeGridIndex(next->y, config_.xy_resolution);
            next->grid_theta = ComputeGridIndex(next->theta, config_.theta_resolution);

            // ----------------------------------------------------
            // 代价结算 (G Cost)
            // ----------------------------------------------------
            double step_penalty = (dir > 0) ? config_.forward_penalty : config_.backward_penalty;
            double steer_cost = std::abs(steer) * config_.steer_penalty;
            double steer_change_cost = std::abs(steer - current->steer) * config_.steer_change_penalty;
            // 换挡惩罚 (前进切倒车，或倒车切前进)
            double gear_switch_cost = (current->is_forward != (dir > 0)) ? 2.0 : 0.0; 

            next->g_cost = current->g_cost + (config_.step_size * step_penalty) + steer_cost + steer_change_cost + gear_switch_cost;
            
            // 记录轨迹状态
            next->steer = steer;
            next->is_forward = (dir > 0);
            next->parent = current;

            expanded_nodes.push_back(next);
        }
    }
    return expanded_nodes;
}

bool HybridAStar::IsCollisionFree(double x, double y, double theta) const {
    // TODO: 接入实际的栅格地图 / Costmap 2D
    // 1. 根据 x, y 查网格
    // 2. 根据 theta 和车辆长宽，使用包围盒计算或者多圆盘 (Circle approximation) 进行碰撞检测
    return true; 
}

double HybridAStar::CalculateHeuristic(const HybridAStarNode::ptr& node, 
                                       double goal_x, double goal_y, double goal_theta) const {
    // TODO: 完整的 Hybrid A* 需要取两者的最大值 (max(H_holonomic, H_non_holonomic))
    // 1. Non-holonomic (无障碍物)：Reed-Shepp 曲线长度
    // 2. Holonomic (有障碍物)：2D A* 在网格地图上算出的到终点距离
    
    // 目前降级为简单的欧拉直线距离作为骨架打底
    return std::hypot(node->x - goal_x, node->y - goal_y);
}

} // namespace autodrive_garage::planning