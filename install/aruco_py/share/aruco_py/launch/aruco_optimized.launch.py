from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
import os

def generate_launch_description():
    # Declare arguments
    camera_config_arg = DeclareLaunchArgument(
        'camera_config',
        default_value='camera_c920.yaml',
        description='Camera config file (camera_c920.yaml, camera_c505.yaml, or camera_720p.yaml)'
    )
    
    device_arg = DeclareLaunchArgument(
        'device',
        default_value='/dev/video0',
        description='Camera device path'
    )
    
    # Get config file path
    config_file = PathJoinSubstitution([
        FindPackageShare('aruco_py'),
        'config',
        LaunchConfiguration('camera_config')
    ])
    
    # Camera node
    camera_node = Node(
        package='v4l2_camera',
        executable='v4l2_camera_node',
        name='camera',
        parameters=[config_file],
        output='screen'
    )
    
    # ArUco detection node
    aruco_node = Node(
        package='aruco_py',
        executable='aruco_node',
        name='aruco_node',
        output='screen'
    )
    
    return LaunchDescription([
        camera_config_arg,
        device_arg,
        camera_node,
        aruco_node,
    ])
