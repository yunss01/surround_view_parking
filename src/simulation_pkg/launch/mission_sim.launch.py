#!/usr/bin/env python3

import os
import subprocess
from launch_ros.actions import Node
from launch import LaunchDescription
from launch.actions import ExecuteProcess
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():

    subprocess.run(['killall', 'gzserver'])
    subprocess.run(['killall', 'gzclient'])

    package_dir = get_package_share_directory('simulation_pkg')
    world_file = os.path.join(package_dir, 'worlds', 'track.world') 
    
    return LaunchDescription([
        
        ExecuteProcess(
            cmd=['gazebo', '--verbose', world_file, '-s', 'libgazebo_ros_factory.so'], 
            output='screen'),
        
        # ExecuteProcess(
        #     cmd=['rqt'], 
        #     output='screen'),
        
        # Node(
        #     package='simulation_pkg', 
        #     executable='video_recording_node',
        #     output='screen'
        # ),
        
        Node(
            package='simulation_pkg',
            executable='load_ego_car_node',
            output='screen',
        ),

        Node(
            package='camera_perception_pkg',
            executable='surroundview_node',
            output='screen',
        ),

        # Node(
        #     package='camera_perception_pkg',
        #     executable='surroundview_maker',
        #     output='screen',
        # ),
        
        # Node(
        #     package='simulation_pkg',
        #     executable='load_obstable_car_node',
        #     output='screen',
        # ),

        # Node(
        #     package='simulation_pkg',
        #     executable='load_traffic_light_node',
        #     output='screen',
        # ),

        Node(
            package='camera_perception_pkg', 
            executable='yolov8_node',
            output='screen',
        ),
        
        Node(
            package='debug_pkg',
            executable='yolov8_visualizer_node',
            output='screen',
        ), 

        Node(
            package='camera_perception_pkg', 
            executable='parking_spot_extractor_node',
            output='screen',
        ),

        # Node(
        #     package='debug_pkg',
        #     executable='path_visualizer_node',
        #     output='screen'
        # ),

        # Node(
        #     package='debug_pkg',
        #     executable='image_saver_node',
        #     output='screen'
        # ),

        Node(
            package='decision_making_pkg', 
            executable='sim_parking_planner_node_repeat',
            output='screen'
        ),
       
        Node(
            package='decision_making_pkg', 
            executable='motion_planner_node',
            output='screen'
        ),
       
        Node(
            package='simulation_pkg', 
            executable='sim_simulation_sender_node',
            output='screen'
        ),

        # [추가] Gazebo에 직접 경로를 그리는 최종 노드
        # Node(
        #     package='debug_pkg',
        #     executable='path_drawer_node',
        #     name='path_drawer_node',
        #     output='screen'
        # ),

    ])      
