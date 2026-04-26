#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry, OccupancyGrid
from geometry_msgs.msg import Twist
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped
import math
import time

class DummyCar(Node):
    def __init__(self):
        super().__init__('dummy_car')
        self.odom_pub = self.create_publisher(Odometry, '/odom', 10)
        self.map_pub = self.create_publisher(OccupancyGrid, '/map', 10) # 发一张空地图
        self.cmd_sub = self.create_subscription(Twist, '/cmd_vel', self.cmd_cb, 10)
        self.tf_broadcaster = TransformBroadcaster(self)
        
        self.x, self.y, self.yaw, self.v = 0.0, 0.0, 0.0, 0.0
        self.last_time = time.time()
        self.timer = self.create_timer(0.02, self.update_physics) # 50Hz 底盘
        self.map_timer = self.create_timer(1.0, self.publish_map) # 1Hz 发地图
        
    def cmd_cb(self, msg):
        self.v = msg.linear.x
        self.yaw_rate = msg.angular.z

    def update_physics(self):
        now = time.time()
        dt = now - self.last_time
        self.last_time = now
        
        # 极简运动学积分
        if hasattr(self, 'yaw_rate'):
            self.yaw += self.yaw_rate * dt
        self.x += self.v * math.cos(self.yaw) * dt
        self.y += self.v * math.sin(self.yaw) * dt
        
        # 发布 Odom
        odom = Odometry()
        odom.header.stamp = self.get_clock().now().to_msg()
        odom.header.frame_id = "map"
        odom.child_frame_id = "base_link"
        odom.pose.pose.position.x = self.x
        odom.pose.pose.position.y = self.y
        odom.pose.pose.orientation.z = math.sin(self.yaw / 2.0)
        odom.pose.pose.orientation.w = math.cos(self.yaw / 2.0)
        odom.twist.twist.linear.x = self.v
        self.odom_pub.publish(odom)

        # 发布 TF (map -> base_link)
        t = TransformStamped()
        t.header.stamp = odom.header.stamp
        t.header.frame_id = 'map'
        t.child_frame_id = 'base_link'
        t.transform.translation.x = self.x
        t.transform.translation.y = self.y
        t.transform.rotation = odom.pose.pose.orientation
        self.tf_broadcaster.sendTransform(t)

    def publish_map(self):
        grid = OccupancyGrid()
        grid.header.stamp = self.get_clock().now().to_msg()
        grid.header.frame_id = "map"
        grid.info.width = 500
        grid.info.height = 500
        grid.info.origin.position.x = -50.0  # 往后退 50 米
        grid.info.origin.position.y = -50.0  # 往后退 50 米
        grid.data = [0] * (500 * 500)       # 生成 250000 个安全区
        self.map_pub.publish(grid)

def main():
    rclpy.init()
    rclpy.spin(DummyCar())
    rclpy.shutdown()

if __name__ == '__main__':
    main()