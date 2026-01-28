import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

# --------------- 설정 구간 ---------------
# 카메라 디바이스 번호 (ls /dev/video* 로 확인 필요)
# 보통 순서대로 0, 2, 4, 6 등으로 잡히거나 0, 1, 2, 3으로 잡힙니다.
# 실제 꽂아보면서 어느 번호가 앞/뒤/좌/우 인지 확인해서 매핑해야 합니다.
CAM_ID_FRONT = 0
CAM_ID_BACK  = 2
CAM_ID_LEFT  = 4
CAM_ID_RIGHT = 6

# 발행할 토픽 이름 (surroundview_node에서 받는 이름과 일치시켜야 함)
TOPIC_FRONT = '/fo_camera/image_raw'
TOPIC_BACK  = '/rear_camera/image_raw'
TOPIC_LEFT  = '/left_camera/image_raw'
TOPIC_RIGHT = '/right_camera/image_raw'

# 해상도 설정 (너무 크면 렉 걸림, 640x480 권장)
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS = 30
# -----------------------------------------

class MultiCameraPublisher(Node):
    def __init__(self):
        super().__init__('multi_camera_publisher_node')
        self.bridge = CvBridge()

        # 1. Publisher 생성
        self.pub_front = self.create_publisher(Image, TOPIC_FRONT, 10)
        self.pub_back  = self.create_publisher(Image, TOPIC_BACK, 10)
        self.pub_left  = self.create_publisher(Image, TOPIC_LEFT, 10)
        self.pub_right = self.create_publisher(Image, TOPIC_RIGHT, 10)

        # 2. 카메라 객체 초기화
        self.cap_front = self.init_camera(CAM_ID_FRONT, "Front")
        self.cap_back  = self.init_camera(CAM_ID_BACK, "Back")
        self.cap_left  = self.init_camera(CAM_ID_LEFT, "Left")
        self.cap_right = self.init_camera(CAM_ID_RIGHT, "Right")

        # 3. 타이머 설정 (30 FPS)
        self.timer = self.create_timer(1.0 / FPS, self.timer_callback)
        self.get_logger().info(f"📸 4-Channel Camera Publisher Started!")

    def init_camera(self, dev_id, name):
        cap = cv2.VideoCapture(dev_id)
        if not cap.isOpened():
            self.get_logger().error(f"❌ Failed to open {name} Camera (ID: {dev_id})")
            return None
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        # cap.set(cv2.CAP_PROP_FPS, FPS) # 지원되는 경우만 동작
        self.get_logger().info(f"✅ {name} Camera (ID: {dev_id}) Initialized.")
        return cap

    def read_and_publish(self, cap, publisher, name):
        if cap is None: return

        ret, frame = cap.read()
        if ret:
            # ROS 메시지로 변환
            msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = "base_link"
            publisher.publish(msg)
        else:
            self.get_logger().warn(f"⚠️ Failed to read frame from {name} camera")

    def timer_callback(self):
        # 4개의 카메라에서 이미지를 읽어서 각각 발행
        self.read_and_publish(self.cap_front, self.pub_front, "Front")
        self.read_and_publish(self.cap_back,  self.pub_back,  "Back")
        self.read_and_publish(self.cap_left,  self.pub_left,  "Left")
        self.read_and_publish(self.cap_right, self.pub_right, "Right")

def main(args=None):
    rclpy.init(args=args)
    node = MultiCameraPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # 종료 시 자원 해제
        if node.cap_front: node.cap_front.release()
        if node.cap_back: node.cap_back.release()
        if node.cap_left: node.cap_left.release()
        if node.cap_right: node.cap_right.release()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()