
import rclpy 
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor

from tf2_ros import TransformListener, LookupException, ExtrapolationException
from tf2_ros.buffer import Buffer

from std_msgs.msg import Empty
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import TransformStamped
from keyboard_msgs.msg import Key
from calibration_msgs.msg import CalibrationDatapoint

import os
from datetime import datetime



class LLSCalibrationDatacollector(Node):

    def __init__(self) -> None:
        super().__init__('lls_datacollector')

        # TODO parametrize 
        config_path = os.path.join(os.getcwd(), 'src', 'lls_calibration', 'config')
        self.calibration_path = os.path.join(config_path, f'calibration_{datetime.now().isoformat(timespec="seconds")}.csv')


        # Declare parameters with default values
        self.declare_parameter('trigger_topic', '/trigger_move')
        self.declare_parameter('pcl_topic', 'scancontrol_pointcloud')
        self.declare_parameter('laser_on_topic', '/laseron')
        self.declare_parameter('laser_off_topic', '/laseroff')
        self.declare_parameter('calibration_msg_topic', 'calibration_data')
        self.declare_parameter('calibrate_topic', 'calibrate_lls')

        # Retrieve the parameters
        trigger_topic   = self.get_parameter('trigger_topic').get_parameter_value().string_value
        pcl_topic       = self.get_parameter('pcl_topic').get_parameter_value().string_value
        laser_on_topic  = self.get_parameter('laser_on_topic').get_parameter_value().string_value
        laser_off_topic = self.get_parameter('laser_off_topic').get_parameter_value().string_value
        calibration_msg = self.get_parameter('calibration_msg_topic').get_parameter_value().string_value
        calibrate_topic = self.get_parameter('calibrate_topic').get_parameter_value().string_value


        self.measurement_interval = 0.5

        self.move_trigger           = self.create_publisher(Empty, trigger_topic, 1)
        self.laser_on_pub           = self.create_publisher(Empty, laser_on_topic, 1)
        self.laser_off_pub          = self.create_publisher(Empty, laser_off_topic, 1)
        self.calibration_trigger    = self.create_publisher(Empty, calibrate_topic, 1)
        self.calibration_data       = self.create_publisher(CalibrationDatapoint, calibration_msg, 1)

        self.pcl_subscriber         = self.create_subscription(PointCloud2, pcl_topic, self.update_pointcloud, 1)
        self.keyboard_listener      = self.create_subscription(Key, 'keydown', self.key_callback, 1)
    
        # Holds pointclouds - is emptied when measure interval is used
        self.pointcloud_buffer = []

        # Track coordinate transformations
        self.tf_buffer = Buffer(cache_time=rclpy.time.Time(seconds=3))
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.global_frame_id = 'base_link'
        self.target_frame_id = 'flange'
        
        self.arcs: list[tuple[PointCloud2, TransformStamped]] = []

        self.sphere_radius = 0.145 / 2 

    def update_pointcloud(self, msg):
        # Update pointcloud and transform to flange
                
        # If the PCL message is empty 
        if len(msg.data) < 1:
            return

        # Lookup the transformation for the time the pointcloud was created
        time = msg.header.stamp
        try:
            tf_trans = self.tf_buffer.lookup_transform(self.global_frame_id, self.target_frame_id, time, timeout=rclpy.duration.Duration(seconds=3))
        except (LookupException, ExtrapolationException):
            self.get_logger().info(f"Could not lookup transform for time: {time}")
            return

        self.pointcloud_buffer = [(msg, tf_trans)]

    def key_callback(self, msg):
        if msg.code == Key.KEY_G:
            self.measure_interval()
        if msg.code == Key.KEY_H:
            self.calibration_trigger.publish(Empty())

    def measure_interval(self):
        # Average stored pointclouds in buffer 
        # avg_pointcloud_arc  = ...
        # tf_transform        = ...
        self.pointcloud_buffer = []
        self.laser_on_pub.publish(Empty())
        self.timer = self.create_timer(self.measurement_interval, self.timer_callback)
        
    def timer_callback(self):
        self.laser_off_pub.publish(Empty())
        self.arcs.extend(self.pointcloud_buffer)

        for pointcloud, transform in self.pointcloud_buffer:
            datapoint = CalibrationDatapoint(pointcloud=pointcloud,
                                             transform=transform)
            self.calibration_data.publish(datapoint)

        self.pointcloud_buffer = []

        self.move_trigger.publish(Empty())
        self.timer.cancel()


def main(args=None):
    rclpy.init(args=args)

    calibrator = LLSCalibrationDatacollector()
    executor = MultiThreadedExecutor()

    rclpy.spin(calibrator, executor=executor)
    calibrator.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
    