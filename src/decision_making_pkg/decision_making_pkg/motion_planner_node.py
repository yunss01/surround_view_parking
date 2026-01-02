import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSDurabilityPolicy, QoSReliabilityPolicy
from interfaces_pkg.msg import PathPlanningResult, MotionCommand
import time

#---------------Variable Setting---------------
SUB_PARKING_PATH_TOPIC_NAME = "parking_path"
PUB_TOPIC_NAME = "topic_control_signal"

TIMER = 0.1
MAX_STEERING = 7.0 
#----------------------------------------------

class MotionPlanningNode(Node):
    def __init__(self):
        super().__init__('motion_planner_node')

        # 파라미터 설정
        self.sub_parking_path_topic = self.declare_parameter('sub_parking_path_topic', SUB_PARKING_PATH_TOPIC_NAME).value
        self.pub_topic = self.declare_parameter('pub_topic', PUB_TOPIC_NAME).value
        self.timer_period = self.declare_parameter('timer', TIMER).value

        # QoS 설정
        self.qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1
        )

        # 데이터 변수
        self.parking_path_data = None
        self.last_parking_cmd_time = 0.0
        
        # 명령값 저장 변수
        self.steering_command = 0
        self.speed_command = 0

        # [심플해진 구성] 오직 주차 경로만 구독하고, 모션 명령만 발행함
        self.parking_path_sub = self.create_subscription(PathPlanningResult, self.sub_parking_path_topic, self.parking_path_callback, self.qos_profile)
        self.publisher = self.create_publisher(MotionCommand, self.pub_topic, self.qos_profile)
        self.timer = self.create_timer(self.timer_period, self.timer_callback)
        
        self.get_logger().info("✅ Simple Motion Planner (Parking Only) Started.")

    def parking_path_callback(self, msg: PathPlanningResult):
        # 주차 플래너로부터 데이터 수신
        if len(msg.x_points) > 0 and len(msg.y_points) > 0:
            self.steering_command = msg.x_points[0] # 조향
            self.speed_command = msg.y_points[0]    # 속도
            self.last_parking_cmd_time = time.time() # 수신 시간 갱신

    def timer_callback(self):
        current_time = time.time()
        
        motion_msg = MotionCommand()

        # [안전장치] 마지막 명령 수신 후 2.0초가 지나지 않았다면 명령 수행
        if (current_time - self.last_parking_cmd_time < 2.0):
            # 조향각 제한 (Clamping)
            final_steer = max(-MAX_STEERING, min(MAX_STEERING, self.steering_command))
            
            motion_msg.steering = int(final_steer)
            motion_msg.left_speed = int(self.speed_command)
            motion_msg.right_speed = int(self.speed_command)
            
            self.get_logger().info(f"Parking Action -> Steer: {motion_msg.steering}, Speed: {motion_msg.left_speed}")
            
        else:
            # 타임아웃 시 정지 (안전)
            motion_msg.steering = 0
            motion_msg.left_speed = 0
            motion_msg.right_speed = 0
            # 로그는 너무 자주 찍히지 않게 조절 가능
            # self.get_logger().warn("No Parking Command! Stopping...", throttle_duration_sec=2)

        self.publisher.publish(motion_msg)

def main(args=None):
    rclpy.init(args=args)
    node = MotionPlanningNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()