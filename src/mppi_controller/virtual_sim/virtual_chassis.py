#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import Twist, PoseStamped
import math
import numpy as np

class VirtualChassis(Node):
    def __init__(self):
        super().__init__('virtual_chassis')
        
        # 订阅 MPPI 发来的控制指令
        self.cmd_sub = self.create_subscription(Twist, '/cmd_vel', self.cmd_callback, 10)
        
        # 发布里程计（当前位置）和 参考路径
        self.odom_pub = self.create_publisher(Odometry, '/odom', 10)
        self.path_pub = self.create_publisher(Path, '/reference_path', 10)
        
        # 车辆初始状态: x, y, yaw, v
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.v = 0.0
        
        # 接收到的最新控制指令
        self.target_v = 0.0
        self.target_yaw_rate = 0.0
        
        # 物理推演定时器 (100Hz，比你的 MPPI 50Hz 更快，模拟真实世界)
        self.dt = 0.01
        self.timer = self.create_timer(self.dt, self.physics_step)
        
        # 生成一条 S 型参考路径
        self.ref_path = self.generate_reference_path()
        self.get_logger().info("🏎️ 虚拟底盘已启动，等待 MPPI 接入...")

    def generate_reference_path(self):
        path = Path()
        path.header.frame_id = "odom"
        for i in range(500):
            s = i * 0.1
            pose = PoseStamped()
            pose.pose.position.x = s
            pose.pose.position.y = 2.0 * math.sin(s / 2.0) # S型弯道
            path.poses.append(pose)
        return path

    def cmd_callback(self, msg):
        self.target_v = msg.linear.x
        self.target_yaw_rate = msg.angular.z

    def physics_step(self):
        # 1. 简单的动力学延迟模拟 (平滑过渡到目标速度)
        self.v += (self.target_v - self.v) * 0.1
        self.yaw += self.target_yaw_rate * self.dt
        
        # 2. 位置更新
        self.x += self.v * math.cos(self.yaw) * self.dt
        self.y += self.v * math.sin(self.yaw) * self.dt
        
        # 3. 发布 Odometry
        odom = Odometry()
        odom.header.stamp = self.get_clock().now().to_msg()
        odom.header.frame_id = "odom"
        odom.pose.pose.position.x = self.x
        odom.pose.pose.position.y = self.y
        # 简化的欧拉角转四元数 (只管 Z 轴偏航)
        odom.pose.pose.orientation.z = math.sin(self.yaw / 2.0)
        odom.pose.pose.orientation.w = math.cos(self.yaw / 2.0)
        odom.twist.twist.linear.x = self.v
        self.odom_pub.publish(odom)
        
        # 4. 发布参考路径
        self.ref_path.header.stamp = self.get_clock().now().to_msg()
        self.path_pub.publish(self.ref_path)

def main():
    rclpy.init()
    node = VirtualChassis()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()