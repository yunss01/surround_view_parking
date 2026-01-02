import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge
import os
import time

class ImageSaverNode(Node):
    def __init__(self):
        super().__init__('image_saver_node')
        self.subscription = self.create_subscription(
            Image,
            '/surround_view/image_raw',
            self.image_callback,
            10)
        self.bridge = CvBridge()
        self.save_dir = os.path.expanduser('~/ros2_saved_images')
        os.makedirs(self.save_dir, exist_ok=True)
        self.get_logger().info(f'이미지 저장 경로: {self.save_dir}')

        self.save_interval = 0.3  
        self.last_save_time = 0
    
    def image_callback(self, msg):
        current_time = time.time()
        
        # 마지막 저장 시간으로부터 설정된 간격(0.1초)이 지났는지 확인
        if current_time - self.last_save_time < self.save_interval:
            return # 간격이 지나지 않았으면 함수 종료

        # 간격이 지났으면, 마지막 저장 시간을 현재 시간으로 업데이트
        self.last_save_time = current_time

        try:
            # ROS 이미지를 OpenCV RGB 이미지로 변환
            cv_image_rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
            
            # OpenCV는 BGR 색상 순서를 사용하므로 정확한 색상 저장을 위해 변환
            cv_image_bgr = cv2.cvtColor(cv_image_rgb, cv2.COLOR_RGB2BGR)
            
            timestamp = int(time.time() * 1000)  # 밀리초 단위 타임스탬프
            filename = os.path.join(self.save_dir, f'image_{timestamp}.png')
            
            # BGR 순서의 이미지 저장
            cv2.imwrite(filename, cv_image_bgr)
            self.get_logger().info(f'이미지 저장 완료: {filename}')
        except Exception as e:
            self.get_logger().error(f'이미지 변환 또는 저장 오류: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = ImageSaverNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
