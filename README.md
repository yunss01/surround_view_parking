[simulation]
ros2 launch simulation_pkg mission_sim.launch.py

[real car]
1. ros2 run camera_perception_pkg image_publisher_node
2. ros2 run camera_perception_pkg surroundview_maker
3. ros2 run camera_perception_pkg yolov8_node
4. ros2 run camera_perception_pkg parking_spot_extractor_node
5. ros2 run decision_making_pkg parking_planner_node
6. ros2 run serial_communication_pkg serial_sender_node
7. ros2 run decision_making_pkg motion_planner_node
