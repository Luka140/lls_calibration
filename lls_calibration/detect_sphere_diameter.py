
import rclpy 
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from std_msgs.msg import Empty

from sensor_msgs.msg import PointCloud2, PointField
from geometry_msgs.msg import TransformStamped, Pose
from calibration_msgs.msg import CalibrationDatapoint

from keyboard_msgs.msg import Key

from tf2_ros import TransformListener, LookupException, ExtrapolationException
from tf2_ros.buffer import Buffer

import copy 
import numpy as np
import open3d as o3d
import pyransac3d as pyrsc

# TODO no need for tf listener
# TODO parameterize 

class RadiusDetector(Node):

    def __init__(self):
        super().__init__('radius_detector')

        self.pcl_listener           = self.create_subscription(PointCloud2, 'scancontrol_pointcloud', self.update_pointcloud, 1)
        self.interval_start_trigger = self.create_subscription(Empty, 'start_radius_measurement', self.start_interval, 1)
        self.interval_end_trigger   = self.create_subscription(Empty, 'end_radius_measurement', self.end_interval, 1)
        self.pcl_publisher          = self.create_publisher(PointCloud2, 'max_radius_profile', 1)

        self.keyboard_listener  = self.create_subscription(Key, 'keydown', self.key_callback, 1)

        laser_on_topic  = '/laseron'
        laser_off_topic = '/laseroff'
        self.laser_on_pub           = self.create_publisher(Empty, laser_on_topic, 1)
        self.laser_off_pub          = self.create_publisher(Empty, laser_off_topic, 1)
        
        trigger_topic   = "/trigger_move"
        self.move_trigger           = self.create_publisher(Empty, trigger_topic, 1)
        
        # Track coordinate transformations
        self.tf_buffer = Buffer(cache_time=rclpy.time.Time(seconds=3))
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.global_frame_id = 'base_link'
    
        self.currently_measuring = False
        self.pointclouds = []

        self.laser_off_pub.publish(Empty())
    
    def update_pointcloud(self, msg):
        # Update pointcloud and transform to flange
        if not self.currently_measuring:
            return 

        # If the PCL message is empty 
        if len(msg.data) < 1:
            return
        self.pointclouds.append(msg)

    def start_interval(self, _):
        self.currently_measuring = True
        self.laser_on_pub.publish(Empty())
        self.move_trigger.publish(Empty())

    def end_interval(self, _):
        self.currently_measuring = False
        self.laser_off_pub.publish(Empty())
        self.move_trigger.publish(Empty())
        self.find_largest_radius(self.pointclouds)
        self.pointclouds = []
        
    
    def find_largest_radius(self, pointclouds):
        if len(pointclouds) < 1:
            return None

        def radius_at_index(i):
            pointcloud = copy.deepcopy(pointclouds[i])
            loaded_array = np.frombuffer(pointcloud.data, dtype=np.float32).reshape(-1, len(pointcloud.fields))[:, :3]

            # Filter the point cloud data
            if not np.any(np.abs(loaded_array) > 0) or not np.any(np.isfinite(loaded_array)) or loaded_array.size == 0:
                return 0

            loaded_array = loaded_array[~np.isnan(loaded_array).any(axis=1)]
            # Filter out any values that are too large to be true 
            loaded_array = loaded_array[~(abs(loaded_array) > 10.).any(axis=1)]

            # Fit a sphere to the filtered data
            sphere_center, axis, sphere_radius, inlier_idx = pyrsc.Circle().fit(loaded_array, thresh=1e-3, maxIteration=1000)
            self.get_logger().info(f"Found radius: {sphere_radius:.4f} for index {i}")
            return sphere_radius if sphere_radius < 0.5 else 0

        # Golden section search parameters
        def golden_search(low, high):
            gr = (np.sqrt(5) - 1) / 2  # Golden ratio

            # Initial points
            c = low
            d = high

            c_prev = -1
            d_prev = -1 

            while abs(high - low) > 2:
                if c != c_prev:
                    radius_c = radius_at_index(int(c))
                if d != d_prev:
                    radius_d = radius_at_index(int(d))

                if radius_c > radius_d:
                    high = d
                else:
                    low = c

                # Recalculate points
                c_prev = c
                d_prev = d 
                
                c = high - gr * (high - low)
                d = low + gr * (high - low)

            return int((low + high) / 2)
        
        # Perform the golden section search on the pointcloud indices
        largest_radius_idx = golden_search(0, len(pointclouds) - 1)
        largest_radius = radius_at_index(largest_radius_idx)

        self.get_logger().info(f"Largest found radius: {largest_radius} - profile idx {largest_radius_idx} out of {len(pointclouds)}")

        pointcloud_publish = pointclouds[largest_radius_idx]
        pointcloud_publish.header.stamp = self.get_clock().now().to_msg()
        self.pcl_publisher.publish(pointcloud_publish)
        return largest_radius, pointclouds[largest_radius_idx]


    # def find_largest_radius(self, pointclouds):

    #     largest_radius = 0.
    #     largest_radius_idx = None

    #     if len(pointclouds) < 1:
    #         return 

    #     for i, pointcloud in enumerate(pointclouds):
                
    #         loaded_array = np.frombuffer(pointcloud.data, dtype=np.float32).reshape(-1, len(pointcloud.fields))[:,:3]

    #         if not np.any(loaded_array > 0):
    #             continue 
    #         if not np.any(np.isfinite(loaded_array)):
    #             continue
    #         if loaded_array.size == 0:
    #             continue

    #         loaded_array = loaded_array[~np.isnan(loaded_array).any(axis=1)] 
    #         loaded_array = loaded_array[~(abs(loaded_array) > 10.).any(axis=1)] 

    #         # TODO maybe downsample the profiles to reduce comp time 
    #         # Or bisect through the list instead of iterating over all 
            
    #         self.get_logger().info(f"Filtering pointcloud {i} out of {len(self.pointclouds)}")
    #         sphere_center, axis, sphere_radius, inlier_idx = pyrsc.Circle().fit(loaded_array, thresh=1e-5, maxIteration=1000)
    #         self.get_logger().info(f'Nr of inliers: {len(inlier_idx)}')
    #         filtered_points = loaded_array[np.array(inlier_idx)]

    #         if sphere_radius > largest_radius and sphere_radius < 0.5:
    #             largest_radius = sphere_radius
    #             largest_radius_idx = i 
    #             self.get_logger().info(f"Found new largest radius: {sphere_radius}")

    #     self.get_logger().info(f"Largest found radius: {largest_radius} - profile idx {largest_radius_idx} out of {len(pointclouds)}")
    #     return largest_radius, pointclouds[largest_radius_idx]
    
    def key_callback(self, msg):
        if msg.code == Key.KEY_V:
            self.start_interval(Empty())
        if msg.code == Key.KEY_B:
            self.end_interval(Empty())

def main(args=None):
    rclpy.init(args=args)

    radius_detector = RadiusDetector()
    executor = MultiThreadedExecutor()

    rclpy.spin(radius_detector, executor=executor)
    radius_detector.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
    