import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge
import numpy as np
import os
import message_filters  # 메시지 동기화를 위한 핵심 라이브러리

class SurroundViewNode(Node):
    def __init__(self):
        super().__init__('surround_view_node')
        
        # --- 최종 결과물의 목표 크기 ---
        self.TARGET_WIDTH = 850
        self.TARGET_HEIGHT = 1150
        
        # ===================================================================
        # ★★★★★ 이 부분의 값들을 직접 수정하며 최적의 값을 찾으세요 ★★★★★
        # -------------------------------------------------------------------
        self.CROP_FRONT = {'top': 0, 'bottom': 200, 'left': 0, 'right': 0}
        self.CROP_BACK  = {'top': 216, 'bottom': 0, 'left': 0, 'right': 0}
        self.CROP_LEFT  = {'top': 455, 'bottom': 485, 'left': 22, 'right': 125}
        self.CROP_RIGHT = {'top': 455, 'bottom': 485, 'left': 143, 'right': 21}

        self.CAR_WIDTH_PX = 150   # 차량 가로 폭 (좌우 카메라 사이 간격)
        self.CAR_LENGTH_PX = 300  # 차량 세로 길이 (앞뒤 카메라 사이 간격)
        # ===================================================================

        self.bridge = CvBridge()

        # --- 1. 구독할 카메라 토픽 정의 ---
        camera_topics = {
            'front': '/fo_camera/image_raw',
            'back':  '/rear_camera/image_raw',
            'left':  '/left_camera/image_raw',
            'right': '/right_camera/image_raw'
        }
        
        # --- 2. 최종 이미지를 발행할 퍼블리셔 생성 ---
        self.publisher_ = self.create_publisher(Image, '/surround_view/image_raw', 10)

        # --- 3. 각 토픽에 대한 message_filters 구독자 생성 ---
        front_sub = message_filters.Subscriber(self, Image, camera_topics['front'])
        back_sub  = message_filters.Subscriber(self, Image, camera_topics['back'])
        left_sub  = message_filters.Subscriber(self, Image, camera_topics['left'])
        right_sub = message_filters.Subscriber(self, Image, camera_topics['right'])
        
        # --- 4. ApproximateTimeSynchronizer 설정 ---
        self.time_synchronizer = message_filters.ApproximateTimeSynchronizer(
            [front_sub, back_sub, left_sub, right_sub],
            queue_size=10,
            slop=0.2 
        )
        self.time_synchronizer.registerCallback(self.sync_callback)
        
        self.get_logger().info('4개 카메라 토픽 동기화 대기 중...')

    def crop_image(self, image, crop_values):
        if image is None: return None
        h, w = image.shape[:2]
        t, b, l, r = crop_values['top'], crop_values['bottom'], crop_values['left'], crop_values['right']
        if t + b >= h or l + r >= w: return image
        return image[t : h - b, l : w - r]

    def sync_callback(self, front_msg, back_msg, left_msg, right_msg):
        """
        4개 이미지가 동기화되면 호출되어 서라운드 뷰 이미지를 생성하고 발행합니다.
        """
        try:
            # --- 단계 1: ROS 메시지를 OpenCV 이미지로 변환 ---
            front_cv = self.bridge.imgmsg_to_cv2(front_msg, desired_encoding='bgr8')
            back_cv  = self.bridge.imgmsg_to_cv2(back_msg,  desired_encoding='bgr8')
            left_cv  = self.bridge.imgmsg_to_cv2(left_msg,  desired_encoding='bgr8')
            right_cv = self.bridge.imgmsg_to_cv2(right_msg, desired_encoding='bgr8')

            # --- 단계 2: 각 이미지 회전 ---
            front_rot = cv2.rotate(front_cv, cv2.ROTATE_180)
            back_rot  = cv2.rotate(back_cv,  cv2.ROTATE_180)
            left_rot  = cv2.rotate(left_cv,  cv2.ROTATE_90_COUNTERCLOCKWISE)
            right_rot = cv2.rotate(right_cv, cv2.ROTATE_90_COUNTERCLOCKWISE) 

            # --- 단계 3: 회전된 이미지 크롭 ---
            front_roi = self.crop_image(front_rot, self.CROP_FRONT)
            back_roi  = self.crop_image(back_rot,  self.CROP_BACK)
            left_roi  = self.crop_image(left_rot,  self.CROP_LEFT)
            right_roi = self.crop_image(right_rot, self.CROP_RIGHT)

            # --- 단계 4: 스티칭(Stitching) ---
            Hf, Wf = front_roi.shape[:2]
            Hb, Wb = back_roi.shape[:2]
            Hl, Wl = left_roi.shape[:2]
            Hr, Wr = right_roi.shape[:2]

            # 중간 영역(Middle Section)의 높이는 '좌측 이미지 높이', '우측 이미지 높이', '설정한 차량 길이' 중 가장 큰 값으로 설정
            mid_height = max(Hl, Hr, self.CAR_LENGTH_PX)

            # 전체 캔버스 크기 계산
            # 너비: 좌측 이미지 폭 + 차량 폭(빈 공간) + 우측 이미지 폭 (단, 앞/뒤 이미지가 더 넓으면 그것 기준)
            canvas_w = max(Wf, Wb, Wl + self.CAR_WIDTH_PX + Wr)
            # 높이: 앞 이미지 + 중간 영역(차량 길이) + 뒤 이미지
            canvas_h = Hf + mid_height + Hb
            
            # 검은색 배경 생성
            canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

            # --- 이미지 배치 ---
            
            # 1) Front (상단 중앙)
            x_offset_f = (canvas_w - Wf) // 2
            canvas[0:Hf, x_offset_f : x_offset_f + Wf] = front_roi
            
            # 2) Back (하단 중앙)
            y_offset_b = Hf + mid_height
            x_offset_b = (canvas_w - Wb) // 2
            canvas[y_offset_b : y_offset_b + Hb, x_offset_b : x_offset_b + Wb] = back_roi
            
            # 3) Left (중단 좌측 - 차량 박스 기준 왼쪽)
            # 중간 영역 내에서 수직 중앙 정렬
            y_offset_mid_l = Hf + (mid_height - Hl) // 2
            canvas[y_offset_mid_l : y_offset_mid_l + Hl, 0 : Wl] = left_roi
            
            # 4) Right (중단 우측 - 차량 박스 기준 오른쪽)
            # 중간 영역 내에서 수직 중앙 정렬
            y_offset_mid_r = Hf + (mid_height - Hr) // 2
            canvas[y_offset_mid_r : y_offset_mid_r + Hr, canvas_w - Wr : canvas_w] = right_roi

            # 단계 5: 최종 결과물 자르기 (Center Crop)
            final_w = self.TARGET_WIDTH
            final_h = self.TARGET_HEIGHT
            
            # 캔버스 중심 좌표
            cy, cx = canvas.shape[:2]
            cy //= 2
            cx //= 2
            
            # 자를 영역 계산 (중심 기준)
            start_x = max(0, cx - final_w // 2)
            end_x   = min(canvas.shape[1], cx + final_w // 2)
            start_y = max(0, cy - final_h // 2)
            end_y   = min(canvas.shape[0], cy + final_h // 2)
            
            # 자르기 실행
            final_canvas = canvas[start_y:end_y, start_x:end_x]

            # --- 단계 6: 최종 이미지를 ROS 메시지로 변환하여 발행 ---
            final_msg = self.bridge.cv2_to_imgmsg(final_canvas, "bgr8")
            final_msg.header = front_msg.header # 타임스탬프 동기화
            self.publisher_.publish(final_msg)

            # 창 속성을 변경하여 마우스로 크기 조절 가능하게 함
            # window_name = "Real-time Surround View"
            # cv2.namedWindow(window_name, cv2.WINDOW_NORMAL) # 창 크기 조절 허용
            # cv2.imshow(window_name, final_canvas)
            # cv2.waitKey(1) # GUI 이벤트를 처리하기 위해 짧게 대기

        except Exception as e:
            self.get_logger().error(f'서라운드 뷰 생성 중 오류 발생: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = SurroundViewNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('사용자에 의해 노드 종료')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()