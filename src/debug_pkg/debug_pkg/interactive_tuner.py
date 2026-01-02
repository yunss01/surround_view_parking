# interactive_tuner.py

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

# --- 설정 변수 ---
# 구독할 이미지 토픽 이름 (multiplexer의 출력 토픽)
IMAGE_TOPIC = 'front120_camera/image_raw' 

# 초기 사다리꼴 좌표 (yolov8_visualizer_node.py의 'front' 값과 동일하게 시작)
INITIAL_POINTS = [ [238, 315], [402, 315], [493, 476], [147, 476] ]
# -----------------


class InteractiveTuner(Node):
    def __init__(self):
        super().__init__('interactive_tuner')
        self.bridge = CvBridge()
        
        # 이미지 토픽 구독자 생성
        self.subscription = self.create_subscription(
            Image,
            IMAGE_TOPIC,
            self.image_callback,
            10)
        
        self.latest_frame = None
        self.last_printed_pts = []

        # OpenCV 윈도우 생성
        self.window_name = 'Camera Feed'
        self.controls_window_name = 'Controls'
        cv2.namedWindow(self.window_name)
        cv2.namedWindow(self.controls_window_name)

        # 이미지 크기에 맞는 트랙바 최대값 설정 (일단 640x480으로 가정)
        img_width, img_height = 640, 480

        # 4개 꼭짓점(P1~P4)의 X, Y 좌표를 위한 8개 트랙바 생성
        for i in range(4):
            # i+1번째 점의 X 좌표 트랙바
            cv2.createTrackbar(f'P{i+1}_X', self.controls_window_name, INITIAL_POINTS[i][0], img_width, self.nothing)
            # i+1번째 점의 Y 좌표 트랙바
            cv2.createTrackbar(f'P{i+1}_Y', self.controls_window_name, INITIAL_POINTS[i][1], img_height, self.nothing)
            
        self.get_logger().info('✅ Interactive Tuner 노드가 시작되었습니다.')
        self.get_logger().info(f"'{IMAGE_TOPIC}' 토픽을 구독합니다.")
        
        # 0.03초마다 화면을 갱신하고 값을 체크하는 타이머 생성
        self.timer = self.create_timer(0.03, self.update_display)

    def nothing(self, x):
        # 트랙바 콜백 함수는 비워둬도 됩니다.
        pass

    def image_callback(self, msg):
        # 받은 이미지 메시지를 OpenCV 프레임으로 변환하여 저장
        self.latest_frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')

    def update_display(self):
        # 최신 프레임이 없으면 아무것도 하지 않음
        if self.latest_frame is None:
            return

        # 그림을 그리기 위해 원본 이미지를 복사
        display_image = self.latest_frame.copy()

        # 현재 트랙바 위치에서 8개 좌표값을 모두 읽어오기
        points = []
        for i in range(4):
            x = cv2.getTrackbarPos(f'P{i+1}_X', self.controls_window_name)
            y = cv2.getTrackbarPos(f'P{i+1}_Y', self.controls_window_name)
            points.append([x, y])

        # 좌표값이 변경되었을 경우에만 터미널에 새로 출력
        if points != self.last_printed_pts:
            self.get_logger().info('새로운 좌표:', throttle_duration_sec=1)
            print(f'points = {points}')
            self.last_printed_pts = points

        # 읽어온 좌표로 사다리꼴 그리기
        pts_np = np.array(points, np.int32).reshape((-1, 1, 2))
        cv2.polylines(display_image, [pts_np], isClosed=True, color=(0, 255, 255), thickness=2)

        # 화면에 결과 이미지 표시
        cv2.imshow(self.window_name, display_image)
        cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    tuner_node = InteractiveTuner()
    rclpy.spin(tuner_node)
    
    tuner_node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
