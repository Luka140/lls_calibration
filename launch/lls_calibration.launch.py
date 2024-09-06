from launch import LaunchDescription
from launch_ros.actions import Node
import launch.actions
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():

    pkg = "lls_calibration"

    data_collector = Node(
        package=pkg,
        executable="lls_calibration",
    )

    calibrator = Node(
        package=pkg,
        executable="calibrate_lls",
    )
    
    return LaunchDescription([data_collector,
                              calibrator])