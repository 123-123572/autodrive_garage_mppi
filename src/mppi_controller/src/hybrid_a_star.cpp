#include "mppi_controller/hybrid_a_star.hpp"
#include "dubins.h"
#include <iostream>
#include <algorithm>
#include <unordered_map>

namespace autodrive_garage::planning {

HybridAStar::HybridAStar(const HybridAStarConfig& config) : config_(config) {
    // 1. 预计算圆的半径 R
    config_.circle_radius = std::hypot(config_.vehicle_length / (2.0 * config_.num_circles), 
                                       config_.vehicle_width / 2.0);

    // 2. 预计算 N 个圆心在车辆局部坐标系下（后轴为原点）的 X 轴偏移量
    config_.circle_offsets.resize(config_.num_circles);
    
    // 车辆最尾部的局部坐标
    double rear_edge_x = - (config_.vehicle_length / 2.0) + config_.rear_axle_to_center; 
    double circle_diameter = config_.vehicle_length / config_.num_circles;

    for (int i = 0; i < config_.num_circles; ++i) {
        // 每个圆覆盖一段长度，圆心在这段长度的中间
        config_.circle_offsets[i] = rear_edge_x + (i + 0.5) * circle_diameter;
    }
    
    std::cout << "碰撞检测 " << config_.num_circles 
              << "圆模型, 半径 R=" << config_.circle_radius << "m" << std::endl;
}

// 接收来自 ROS 节点的 Costmap 数据并存入配置中
void HybridAStar::UpdateMap(const std::vector<uint8_t>& costmap, int width, int height, double origin_x, double origin_y) {
    config_.costmap = costmap;
    config_.map_width = width;
    config_.map_height = height;
    config_.origin_x = origin_x; // 记录真实原点
    config_.origin_y = origin_y; // 记录真实原点
}

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

        // 到达目标点判断 
        if (std::hypot(current->x - goal_x, current->y - goal_y) < config_.xy_resolution) {
            std::cout << "Hybrid A* 找到终点！迭代次数: " << iter_count << std::endl;
            
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

    std::cerr << "Hybrid A* 搜索失败或达到最大迭代次数。" << std::endl;
    return false;
}

std::vector<HybridAStarNode::ptr> HybridAStar::ExpandNode(const HybridAStarNode::ptr& current) {
    std::vector<HybridAStarNode::ptr> expanded_nodes;
    
    // 转向角等分采样 
    double steer_step_val = (2.0 * config_.max_steer) / (config_.steer_step - 1);

    // 两个方向：1 代表前进，-1 代表倒车
    std::vector<int> directions = {1, -1};

    for (int dir : directions) {
        for (int i = 0; i < config_.steer_step; ++i) {
            double steer = -config_.max_steer + i * steer_step_val;

            auto next = std::make_shared<HybridAStarNode>();
            
            // 物理引擎：离散化的运动学推演 (Bicycle Model)
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

            // 代价结算 (G Cost)
            //
            double step_penalty = (dir > 0) ? config_.forward_penalty : config_.backward_penalty;
            //稳态转向代价
            double steer_cost = std::abs(steer) * config_.steer_penalty;
            //动态平顺性代价
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
    // 如果没有地图数据，直接返回 true（或者抛出警告）
    if (config_.costmap.empty() || config_.map_width == 0) {
        return true; 
    }

    // 提取三角函数
    const double cos_theta = std::cos(theta);
    const double sin_theta = std::sin(theta);

    // 遍历预计算好的 N 个圆
    for (int i = 0; i < config_.num_circles; ++i) {
        
        // 1. 计算圆心的全局坐标 (通过旋转平移)
        double cx = x + config_.circle_offsets[i] * cos_theta;
        double cy = y + config_.circle_offsets[i] * sin_theta;

        // 2. 转换到 Costmap 栅格坐标
        int grid_x = ComputeGridIndex(cx - config_.origin_x, config_.xy_resolution);
        int grid_y = ComputeGridIndex(cy - config_.origin_y, config_.xy_resolution);

        // 3. 地图边界硬核保护
        if (grid_x < 0 || grid_x >= config_.map_width || 
            grid_y < 0 || grid_y >= config_.map_height) {
            return false; // 开出地图边界，判定为碰撞！
        }

        // 4. O(1) 极速查表 
        // 外部喂进来的 costmap 必须已经是根据 circle_radius 膨胀过的！
        int index = grid_y * config_.map_width + grid_x;
        
        if (config_.costmap[index] >= config_.lethal_cost) {
            return false; // 圆心踩到雷区，直接驳回该轨迹节点！
        }
    }

    // 4 个圆都没事，一路绿灯
    return true; 
}


double HybridAStar::CalculateHeuristic(const HybridAStarNode::ptr& node, 
                                       double goal_x, double goal_y, double goal_theta) const 
{
    // 1. 当前点
    double x0 = node->x;
    double y0 = node->y;
    double th0 = node->theta;

    // 2. 目标点
    double x1 = goal_x;
    double y1 = goal_y;
    double th1 = goal_theta;

    // 4. 创建 Dubins 路径
DubinsPath path;

// 起点 q0 = [x, y, theta]
double q0[3] = {x0, y0, th0};

// 终点 q1 = [x, y, theta]
double q1[3] = {x1, y1, th1};

// 最小转弯半径
double rho = config_.min_turning_radius;

// 调用
int ret = dubins_shortest_path(&path, q0, q1, rho);

if (ret != 0) {
    // 失败退化为欧几里得
    return std::hypot(x1 - x0, y1 - y0);
}

// Dubins 路径长度
return dubins_path_length(&path);
}

} // namespace autodrive_garage::planning