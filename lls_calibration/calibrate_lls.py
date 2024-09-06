
import rclpy 
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor

from std_msgs.msg import Empty
from sensor_msgs.msg import PointCloud2, PointField
from geometry_msgs.msg import TransformStamped, Pose
from calibration_msgs.msg import CalibrationDatapoint

from visualization_msgs.msg import Marker

import os
import copy
import numpy as np
from datetime import datetime
from scipy.optimize import least_squares, minimize, Bounds
from scipy.spatial.transform import Rotation as R

import open3d as o3d
import pyransac3d as pyrsc

# TODO try with zero initial conditions 
# TODO try with different number of arcs
# TODO parametrize 

class LLSCalibrator(Node):

    def __init__(self) -> None:
        super().__init__('lls_calibrator')

        config_path = os.path.join(os.getcwd(), 'src', 'lls_calibration', 'config')
        self.calibration_path = os.path.join(config_path, f'calibration_{datetime.now().isoformat(timespec="seconds")}.csv')

        calibration_msg_topic       = 'calibration_data'
        calibration_trigger_topic   = 'calibrate_lls'
        self.calibration_subscriber = self.create_subscription(CalibrationDatapoint, calibration_msg_topic, self.update_data, 1)
        self.calibration_trigger    = self.create_subscription(Empty, calibration_trigger_topic, self.calibrate, 1)
        self.pcl_publisher          = self.create_publisher(PointCloud2, 'calibration_clouds', 1)
        self.marker_publisher       = self.create_publisher(Marker, "visualization_marker", 1)
      
        self.global_frame_id = 'base_link'

        self.datapoints: list[tuple[PointCloud2, TransformStamped]] = []

        self.sphere_radius = 0.145 / 2 
        self.ransac_threshold = 1e-4
     
        # XYZW 
        self.approx_rotation = np.array([0., np.sqrt(2)/2, np.sqrt(2)/2, 0.])
        self.approx_offset   = [28.5/1000, 42.01/1000, 32.01/1000]


    def update_data(self, msg):
        # Update pointcloud and transform to flange
        pcl = msg.pointcloud
        tf  = msg.transform 
        self.datapoints.append((pcl, tf))

    def calibrate(self, msg):
        """
        Fit the arcs to a sphere to obtain the transformation from the flange to the LLS 
        """

        approx_rotation_matrix = _get_mat_from_quat(np.array([self.approx_rotation[-1], *self.approx_rotation[:-1]]))


        loaded_arcs: list[o3d.geometry.Pointcloud, np.ndarray] = []
        center_positions = []

        self.get_logger().info("Loading pointclouds and filtering using RANSAC")
        # Preload messages to usable objects 
        i = 0 
        for pointcloud, transform in self.datapoints:
            i += 1 
            # Open all pointcloud messages and create o3d pointclouds 
            loaded_array = np.frombuffer(pointcloud.data, dtype=np.float32).reshape(-1, len(pointcloud.fields))[:,:3]

            if not np.any(loaded_array > 0):
                continue 
            if not np.any(np.isfinite(loaded_array)):
                continue
            if loaded_array.size == 0:
                continue

            loaded_array = loaded_array[~np.isnan(loaded_array).any(axis=1)] 
            loaded_array = loaded_array[~(abs(loaded_array) > 10.).any(axis=1)] 

            self.get_logger().info(f"Filtering pointcloud {i} out of {len(self.datapoints)}")
            sphere_center, axis, sphere_radius, inlier_idx = pyrsc.Circle().fit(loaded_array, thresh=self.ransac_threshold, maxIteration=1000)
            self.get_logger().info(f'Nr of inliers: {len(inlier_idx)}')
            filtered_points = loaded_array[np.array(inlier_idx)]
    
            # Create o3d pointcloud based on pcl message
            o3d_pcl = o3d.geometry.PointCloud()
            o3d_pcl.points = o3d.utility.Vector3dVector(filtered_points)
            
            # Create a transformation matrix from the transformation message obtained from tf_trans
            tf_mat = self.tf_transform_to_matrix(transform)
            loaded_arcs.append((o3d_pcl, tf_mat))
            
            # Crude initial transform to get an initial guess for the center position
            bounding_box_pcl = copy.deepcopy(o3d_pcl)
            transformed_pts = bounding_box_pcl.rotate(approx_rotation_matrix, center=np.array([[0],[0],[0]])).transform(tf_mat)
            bounding_box = transformed_pts.get_axis_aligned_bounding_box()
            center_positions.append(bounding_box.get_center().T)
        
        avg_pos = np.mean(np.array(center_positions), axis=0)
        self.get_logger().info(f"Aprroximate average positions of pointcloud: {avg_pos}")
               
        def fit_midpoint(center_sphere, arcs, sphere_rad, lls_transform):
            rotation_matrix = _get_mat_from_quat(np.array([lls_transform[-1], *lls_transform[3:-1]]))
            flange_lls_transform = np.eye(4)
            flange_lls_transform[:3,:3] = rotation_matrix
            flange_lls_transform[:3, 3] = lls_transform[:3]

            arc_copy = copy.deepcopy(arcs)
            combined_pcl = o3d.geometry.PointCloud()

            for pointcloud, tf_robot in arc_copy:
                tf = tf_robot @ flange_lls_transform
                combined_pcl += pointcloud.transform(tf)
            
            self.pcl_publisher.publish(self.create_pcl_msg(combined_pcl))
            self.marker_publisher.publish(self.create_marker_msg(center_sphere))
            points = np.asarray(combined_pcl.points)
            points -= center_sphere
            radius = (points[:,0] ** 2 + points[:,1] ** 2 + points[:,2] ** 2) ** 0.5
            sum_squared_errors = np.sum(np.square(radius - sphere_rad))
            return sum_squared_errors
        
        def fit_arcs_sphere(x, arcs, sphere_rad):
            """
            Inner function for SciPy optimize to minimize 
            """
            # Create transform matrix from flange to laser based on x
            flange_lls_translation  = x[:3]
            flange_lls_rotation     = x[3:7]
            centre_point_sphere     = x[-3:]

            rotation_matrix = _get_mat_from_quat(np.array([flange_lls_rotation[-1], *flange_lls_rotation[:-1]]))
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
            self.marker_publisher.publish(self.create_marker_msg(centre_point_sphere))
            
            # Translate the combined pcl to center (0,0,0) based on input centerpoint
            points = np.asarray(combined_pcl.points)
            points -= centre_point_sphere
            radius = (points[:,0] ** 2 + points[:,1] ** 2 + points[:,2] ** 2) ** 0.5
            # self.get_logger().info(f"Mean Radius: {np.mean(radius)}")
            sum_squared_errors = np.sum(np.square(radius - sphere_rad))

            return sum_squared_errors

        # Format: translations in xyz, rotations xyz, centerpoint calibration sphere xyz 
        calibration_sphere_coordinate_estimation = avg_pos.flatten()
        initial_guess = np.array([*self.approx_offset, *self.approx_rotation, *calibration_sphere_coordinate_estimation])
        self.get_logger().info(f'Initial guess:"\n  Translation: {initial_guess[:3]}\n  Rotation: {initial_guess[3:7]}\n  Center sphere: {initial_guess[-3:]}')

        lb_translation, ub_translation = [-1., -1., -1.], [1., 1., 1.]
        lb_rotation, ub_rotation       = [-1., -1., -1., -1.], [1., 1., 1., 1.] # TODO dunno how well it will work to clamp quaternions like this
        lb_center, ub_center           = [-2., -2., -2.], [2., 2., 2.]
        

        bounds_midpoint_opt = Bounds(lb=lb_center, ub=ub_center)
        bounds_full_opt     = Bounds(lb=[*lb_translation, *lb_rotation, *lb_center], 
                                     ub=[*ub_translation, *ub_rotation, *ub_center])
        

        result = minimize(fit_midpoint, x0=initial_guess[-3:], args=(loaded_arcs, self.sphere_radius, initial_guess[:7]), bounds=bounds_midpoint_opt)
        midpoint_result = result.x

        result = minimize(fit_arcs_sphere, x0=[*initial_guess[:7], *midpoint_result], args=(loaded_arcs, self.sphere_radius), tol=1e-12, bounds=bounds_full_opt)
        
        if result.success:
            self.get_logger().info(f'\nCalibration successfull.\nReason for termination: {result.message}')
        else:
            self.get_logger().info(f'\nCalibration failed.\nReason for termination: {result.message}')
        
        with open(self.calibration_path, 'w') as f:
            f.write('offset_x, offset_y, offset_z,rotation_x,rotation_y,rotation_z,rotation_w,sphere_x,sphere_y,sphere_z\n')
            f.write(f"{','.join([str(res) for res in result.x])}\n")
        self.get_logger().info(f'Result stored at {self.calibration_path}')
        
        

        self.marker_publisher.publish(self.create_marker_msg(result.x[-3:]))
        return
    
    def tf_transform_to_matrix(self, tf_trans) -> np.ndarray:
        transform = np.eye(4)
        rot = tf_trans.transform.rotation
        trans = tf_trans.transform.translation

        # If this is altered - be careful of the order of the quaternion (w,x,y,z) vs (x,y,z,w)
        # Some packages use one, some use the other and they often don't specify which is used.  
        transform[:3,:3] = _get_mat_from_quat(np.array([rot.w, rot.x, rot.y, rot.z]))
        transform[:3,3] = [trans.x, trans.y, trans.z]
        return transform
    
    def create_marker_msg(self, position):
        sphere_position = Pose()
        sphere_position.position.x, sphere_position.position.y, sphere_position.position.z = position
        marker = Marker(type=Marker.SPHERE, action=0, pose=sphere_position)
        marker.header.frame_id = self.global_frame_id
        marker.scale.x = self.sphere_radius * 2
        marker.scale.y = self.sphere_radius * 2
        marker.scale.z = self.sphere_radius * 2
        marker.color.a = 1.
        marker.color.g = 1.
        return marker

    
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
    