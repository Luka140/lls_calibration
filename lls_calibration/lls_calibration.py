
import rclpy 
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor

from tf2_ros import TransformListener, LookupException, ExtrapolationException
from tf2_ros.buffer import Buffer

from std_msgs.msg import Empty
from sensor_msgs.msg import PointCloud2, PointField
from geometry_msgs.msg import TransformStamped, Pose
from keyboard_msgs.msg import Key
from visualization_msgs.msg import Marker

import os
import copy
import numpy as np
from datetime import datetime
from scipy.optimize import least_squares, minimize, Bounds
from scipy.spatial.transform import Rotation as R

import open3d as o3d


class LLSCalibrator(Node):

    def __init__(self) -> None:
        super().__init__('lls_calibrator')

        # TODO parametrize 
        config_path = os.path.join(os.getcwd(), 'src', 'lls_calibration', 'config')
        self.calibration_path = os.path.join(config_path, f'calibration_{datetime.now().isoformat(timespec="seconds")}.csv')

        controller_name = 'ur_script'
        trigger_topic   = f"/{controller_name}/trigger_move"
        pcl_topic       = 'scancontrol_pointcloud'
        laser_on_topic  = '/laseron'
        laser_off_topic = '/laseroff'

        self.measurement_interval = 0.5

        self.move_trigger       = self.create_publisher(Empty, trigger_topic, 1)
        self.laser_on_pub       = self.create_publisher(Empty, laser_on_topic, 1)
        self.laser_off_pub      = self.create_publisher(Empty, laser_off_topic, 1)
        self.pcl_subscriber     = self.create_subscription(PointCloud2, pcl_topic, self.update_pointcloud, 1)
        self.pcl_publisher      = self.create_publisher(PointCloud2, 'calibration_clouds', 1)
        self.keyboard_listener  = self.create_subscription(Key, 'keydown', self.key_callback, 1)
        self.marker_publisher   = self.create_publisher(Marker, "visualization_marker", 1)

        # Holds pointclouds - is emptied when measure interval is used
        self.pointcloud_buffer = []

        # Track coordinate transformations
        self.tf_buffer = Buffer(cache_time=rclpy.time.Time(seconds=3))
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.global_frame_id = 'base_link'
        self.target_frame_id = 'flange'
        
        self.arcs: list[tuple[PointCloud2, TransformStamped]] = []

        self.sphere_radius = 0.145 / 2 

        box_points = np.array([[-10,-10,-10],
                               [-10, -10, 10],
                               [10, 10, -10],
                               [10,10,10]])
        self.crop_box =  o3d.geometry.AxisAlignedBoundingBox.create_from_points(
            o3d.utility.Vector3dVector(box_points))


    def update_pointcloud(self, msg):
        # Update pointcloud and transform TO FLANGE NOT SCANNER
                
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
            self.calibrate()

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
        self.pointcloud_buffer = []

        self.move_trigger.publish(Empty())
        self.timer.cancel()

    def calibrate(self):
        """
        Fit the arcs to a sphere to obtain the transformation from the flange to the LLS 
        """

        offset                                      = [28.5/1000, 42.01/1000, 32.01/1000]
        rotation                                    = [1.570796327, -3.141592654, 0]
        # rotation                                    = [0.0, 3.141592654, -1.570796327]
        # r = R.from_rotvec(rotation)
        # r = R.from_euler('zxy', [rotation[-1], *rotation[:-1]])
        # approx_rotation_matrix = r.as_matrix()
        approx_rotation_matrix = o3d.geometry.get_rotation_matrix_from_xyz(rotation)
        
        ##############################   FOR DEBUGGING PURPOSES ONLY     ############################
        try:
            lookup_time = self.get_clock().now()
            transf = self.tf_buffer.lookup_transform('flange', 'scancontrol', lookup_time, timeout=rclpy.duration.Duration(seconds=3))
            trans_mat = self.tf_transform_to_matrix(transf)
            approx_rotation_matrix = trans_mat[:3,:3]
            self.get_logger().info(f'Transform matrix static tf: {trans_mat}')
        except (LookupException, ExtrapolationException):
            self.get_logger().info(f"Could not lookup transform for time: {time}")

        #############################################################################################

        loaded_arcs: list[o3d.geometry.Pointcloud, np.ndarray] = []
        center_positions = []

        # Preload messages to usable objects 
        for pointcloud, transform in self.arcs:
            # Open all pointcloud messages and create o3d pointclouds 
            loaded_array = np.frombuffer(pointcloud.data, dtype=np.float32).reshape(-1, 4)
            loaded_array = loaded_array[~np.isnan(loaded_array).any(axis=1)] 
            if not np.any(loaded_array > 0):
                continue 
            if not np.any(np.isfinite(loaded_array)):
                continue
            if loaded_array.size == 0:
                continue
            
            # Create and clean up o3d pointcloud based on pcl message
            o3d_pcl = o3d.geometry.PointCloud()
            o3d_pcl.points = o3d.utility.Vector3dVector(loaded_array[:,:3])
            o3d_pcl, _removed_indices = o3d_pcl.remove_non_finite_points().remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
            o3d_pcl = o3d_pcl.crop(self.crop_box)
            
            # Create a transformation matrix from the transformation message obtained from tf_trans
            tf_mat = self.tf_transform_to_matrix(transform)
            loaded_arcs.append((o3d_pcl, tf_mat))
            
            # Crude initial transform to get an initial guess for the center position
            # self.get_logger().info(f'rotation mat: {approx_rotation_matrix}')
            bounding_box_pcl = copy.deepcopy(o3d_pcl)
            transformed_pts = bounding_box_pcl.rotate(approx_rotation_matrix, center=np.array([[0],[0],[0]])).transform(tf_mat)
            bounding_box = transformed_pts.get_axis_aligned_bounding_box()
            center_positions.append(bounding_box.get_center().T)
        
        avg_pos = np.mean(np.array(center_positions), axis=0)
        self.get_logger().info(f"Aprroximate average positions of pointcloud: {avg_pos}")
               
        def fit_midpoint(center_sphere, arcs, sphere_rad, lls_transform):
            rotation_matrix = o3d.geometry.get_rotation_matrix_from_xyz(lls_transform[3:])
            flange_lls_transform = np.eye(4)
            flange_lls_transform[:3,:3] = rotation_matrix
            flange_lls_transform[:3, 3] = lls_transform[:3]

            arc_copy = copy.deepcopy(arcs)
            combined_pcl = o3d.geometry.PointCloud()

            for pointcloud, tf_robot in arc_copy:
                tf = tf_robot @ flange_lls_transform
                combined_pcl += pointcloud.transform(tf)
            
            self.pcl_publisher.publish(self.create_pcl_msg(combined_pcl))

            points = np.asarray(combined_pcl.points)
            points -= center_sphere
            radius = (points[:,0] ** 2 + points[:,1] ** 2 + points[:,2] ** 2) ** 0.5
            self.get_logger().info(f"New center position {center_sphere}")
            self.get_logger().info(f"Radii: {radius}")
            sum_squared_errors = np.sum(np.square(radius - sphere_rad))
            return sum_squared_errors
        
        def fit_arcs_sphere(x, arcs, sphere_rad):
            """
            Inner function for SciPy optimize to minimize 
            """
            # Create transform matrix from flange to laser based on x
            flange_lls_translation  = x[:3]
            flange_lls_rotation     = x[3:6]
            centre_point_sphere     = x[-3:]

            rotation_matrix = o3d.geometry.get_rotation_matrix_from_xyz(flange_lls_rotation)
            # self.get_logger().info(f'Rotation mat: {rotation_matrix}')
            flange_lls_transform = np.eye(4)
            flange_lls_transform[:3,:3] = rotation_matrix
            flange_lls_transform[:3, 3] = flange_lls_translation

            combined_pcl = o3d.geometry.PointCloud()

            arc_copy = copy.deepcopy(arcs)
            for pointcloud, transform in arc_copy:
                tf = transform @ flange_lls_transform
                # combined_pcl += pointcloud.transform(tf)
                combined_pcl += pointcloud.transform(flange_lls_transform).transform(transform)

            self.pcl_publisher.publish(self.create_pcl_msg(combined_pcl))
            
            # Translate the combined pcl to center (0,0,0) based on input centerpoint
            points = np.asarray(combined_pcl.points)
            points -= centre_point_sphere
            radius = (points[:,0] ** 2 + points[:,1] ** 2 + points[:,2] ** 2) ** 0.5
            # self.get_logger().info(f"Radii: {radius}")
            sum_squared_errors = np.sum(np.square(radius - sphere_rad))

            return sum_squared_errors

        # Format: translations in xyz, rotations xyz, centerpoint calibration sphere xyz 
        calibration_sphere_coordinate_estimation = avg_pos.flatten()
        initial_guess = np.array([*offset, *rotation, *calibration_sphere_coordinate_estimation])
        self.get_logger().info(f'Initial guess" {initial_guess}')

        bounds = Bounds(lb=[-1., -1., -1., -1.1*np.pi, -1.1*np.pi, -1.1*np.pi, -2., -2.,-2.],
                        ub=[1., 1., 1., 1.1*np.pi, 1.1*np.pi, 1.1*np.pi, 2., 2.,2.])
        

        result = minimize(fit_midpoint, x0=initial_guess[-3:], args=(loaded_arcs, self.sphere_radius, initial_guess[:6]))
        midpoint_result = result.x

        result = minimize(fit_arcs_sphere, x0=[*initial_guess[:6], *midpoint_result], args=(loaded_arcs, self.sphere_radius), bounds=bounds, tol=1e-12)
        
    
        self.get_logger().info(f'Calibration successfull: {result.success}\nReason for termination: {result.message}')
        
        with open(self.calibration_path, 'w') as f:
            f.write('offset_x, offset_y, offset_z,rotation_x,rotation_y,rotation_z,sphere_x,sphere_y,sphere_z\n')
            f.write(f"{','.join([str(res) for res in result.x])}\n")
        self.get_logger().info(f'Result stored at {self.calibration_path}')
        
        
        sphere_position = Pose()
        sphere_position.position.x, sphere_position.position.y, sphere_position.position.z = result.x[6:]
        # sphere_position.position.x, sphere_position.position.y, sphere_position.position.z = result.x
        marker = Marker(type=Marker.SPHERE, action=0, pose=sphere_position)
        marker.header.frame_id = self.global_frame_id
        marker.scale.x = self.sphere_radius * 2
        marker.scale.y = self.sphere_radius * 2
        marker.scale.z = self.sphere_radius * 2
        marker.color.a = 1.
        marker.color.g = 1.
        self.marker_publisher.publish(marker)
        return
    
    def tf_transform_to_matrix(self, tf_trans) -> np.ndarray:
        transform = np.eye(4)
        rot = tf_trans.transform.rotation
        trans = tf_trans.transform.translation

        # If this is altered - be careful of the order of the quaternion (w,x,y,z) vs (x,y,z,w)
        # Some packages use one, other use the other and they often don't specify which is used.  
        transform[:3,:3] = _get_mat_from_quat(np.array([rot.w, rot.x, rot.y, rot.z]))
        transform[:3,3] = [trans.x, trans.y, trans.z]
        return transform
    
    def create_pcl_msg(self, o3d_pcl):

        datapoints = np.asarray(o3d_pcl.points, dtype=np.float32)

        pointcloud = PointCloud2()
        pointcloud.header.frame_id = self.global_frame_id

        dims = ['x', 'y', 'z']

        bytes_per_point = 4
        fields = [PointField(name=direction, offset=i * bytes_per_point, datatype=PointField.FLOAT32, count=1) for i, direction in enumerate(dims)]
        pointcloud.fields = fields
        pointcloud.point_step = len(fields) * bytes_per_point
        total_points = datapoints.shape[0]
        pointcloud.is_dense = False
        pointcloud.height = 1
        pointcloud.width = total_points

        pointcloud.data = datapoints.flatten().tobytes()
        return pointcloud
    

# TODO replace by scipy function 
def _get_mat_from_quat(quaternion: np.ndarray) -> np.ndarray:
    """
    ===========================================================================================================
    TAKEN FROM THE ROS2 REPO: 
    https://github.com/ros2/geometry2/blob/rolling/tf2_geometry_msgs/src/tf2_geometry_msgs/tf2_geometry_msgs.py
    simply importing it led to issues because it is not in the setup file in humble. 
    ===========================================================================================================

    Convert a quaternion to a rotation matrix.

    This method is based on quat2mat from https://github.com
    f185e866ecccb66c545559bc9f2e19cb5025e0ab/transforms3d/quaternions.py#L101 ,
    since that library is not available via rosdep.

    :param quaternion: A numpy array containing the w, x, y, and z components of the quaternion
    :returns: The rotation matrix
    """
    Nq = np.sum(np.square(quaternion))
    if Nq < np.finfo(np.float64).eps:
        return np.eye(3)

    XYZ = quaternion[1:] * 2.0 / Nq
    wXYZ = XYZ * quaternion[0]
    xXYZ = XYZ * quaternion[1]
    yYZ = XYZ[1:] * quaternion[2]
    zZ = XYZ[2] * quaternion[3]

    return np.array(
        [[1.0-(yYZ[0]+zZ), xXYZ[1]-wXYZ[2], xXYZ[2]+wXYZ[1]],
        [xXYZ[1]+wXYZ[2], 1.0-(xXYZ[0]+zZ), yYZ[1]-wXYZ[0]],
        [xXYZ[2]-wXYZ[1], yYZ[1]+wXYZ[0], 1.0-(xXYZ[0]+yYZ[0])]])


def main(args=None):
    rclpy.init(args=args)

    calibrator = LLSCalibrator()
    executor = MultiThreadedExecutor()

    rclpy.spin(calibrator, executor=executor)
    calibrator.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
    