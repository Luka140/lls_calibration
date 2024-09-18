from launch import LaunchDescription
from launch_ros.actions import Node
import launch.actions
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():

    pkg = "lls_calibration"

    radius_detector = Node(
        package=pkg,
        executable='radius_detector'
    )
    
    return LaunchDescription([radius_detector])