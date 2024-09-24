from launch import LaunchDescription
from launch_ros.actions import Node

import os
from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource



def generate_launch_description():

    pkg = "lls_calibration"

    # path_trajectory = 'trajectory_find_rad.yaml'

    data_collector = Node(
        package=pkg,
        executable="lls_calibration",
    )

    calibrator = Node(
        package=pkg,
        executable="calibrate_lls",
    )

        # Scancontrol driver & calibration
    scanner = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [os.path.join(get_package_share_directory('ur_trajectory_controller'),'launch','ur_launch_scanner.launch.py')])
    )

       
    

    return LaunchDescription([data_collector,
                              calibrator,
                              scanner])