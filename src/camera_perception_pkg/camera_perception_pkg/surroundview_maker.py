import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge
import numpy as np
import os
import message_filters

def nothing(x):
    pass

class SurroundViewNode(Node):
    def __init__(self):
        super().__init__('surround_view_node')
        
        # --- 최종 결과물의 목표 크기 ---
        self.TARGET_WIDTH = 872
        self.TARGET_HEIGHT = 1024

        self.bridge = CvBridge()
        
        # ===================================================================
        # [튜닝용] 초기값 설정
        # ===================================================================
        self.crop_params = {
            'F_Top': 20, 'F_Bot': 0, 'F_Left': 0, 'F_Right': 0,
            'B_Top': 0, 'B_Bot': 20, 'B_Left': 0, 'B_Right': 0,
            'L_Top': 180, 'L_Bot': 230, 'L_Left': 270, 'L_Right': 340,
            'R_Top': 180, 'R_Bot': 230, 'R_Left': 340, 'R_Right': 285
        }
        self.car_params = {
            'Width': 215,
            'Length': 400
        }
        
        # --- [왜곡 보정] 파라미터 초기값 ---
        self.undistort_params = {
            'fx': 300, 'fy': 300,
            'cx': 640, 'cy': 360,   # 1280x720의 중심점 (640, 360)
            'k1': 0, 'k2': 0,
            'p1': 0, 'p2': 0
        }

        # --- 튜닝 윈도우 설정 ---
        cv2.namedWindow('Tuning Controls', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Tuning Controls', 500, 1000) # 슬라이더가 많아져서 창을 키움

        cv2.namedWindow("Surround View Tuning", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Surround View Tuning", 800, 1000)

        self.create_trackbars()

        # ===================================================================

        camera_topics = {
            'front': '/fo_camera/image_raw',
            'back':  '/rear_camera/image_raw',
            'left':  '/left_camera/image_raw',
            'right': '/right_camera/image_raw'
        }
        
        self.publisher_ = self.create_publisher(Image, '/surround_view/image_raw', 10)
        
        front_sub = message_filters.Subscriber(self, Image, camera_topics['front'])
        back_sub  = message_filters.Subscriber(self, Image, camera_topics['back'])
        left_sub  = message_filters.Subscriber(self, Image, camera_topics['left'])
        right_sub = message_filters.Subscriber(self, Image, camera_topics['right'])
        
        self.time_synchronizer = message_filters.ApproximateTimeSynchronizer(
            [front_sub, back_sub, left_sub, right_sub],
            queue_size=10,
            slop=0.2 
        )
        self.time_synchronizer.registerCallback(self.sync_callback)
        self.get_logger().info("실시간 튜닝 모드 시작! (Front/Back 상하좌우 조절 기능 추가됨)")

    def create_trackbars(self):
        # --- Undistort Params ---
        cv2.createTrackbar('K_fx', 'Tuning Controls', self.undistort_params['fx'], 2000, nothing)
        cv2.createTrackbar('K_fy', 'Tuning Controls', self.undistort_params['fy'], 2000, nothing)
        cv2.createTrackbar('D_k1 (+100)', 'Tuning Controls', int((self.undistort_params['k1'] + 1.0) * 100), 200, nothing)
        
        # --- Crop Params ---
        # Front (가로 1280, 세로 720)
        cv2.createTrackbar('F_Top', 'Tuning Controls', self.crop_params['F_Top'], 720, nothing)
        cv2.createTrackbar('F_Bot', 'Tuning Controls', self.crop_params['F_Bot'], 720, nothing)
        cv2.createTrackbar('F_Left', 'Tuning Controls', self.crop_params['F_Left'], 1280, nothing)
        cv2.createTrackbar('F_Right', 'Tuning Controls', self.crop_params['F_Right'], 1280, nothing)
        
        # Back (가로 1280, 세로 720)
        cv2.createTrackbar('B_Top', 'Tuning Controls', self.crop_params['B_Top'], 720, nothing)
        cv2.createTrackbar('B_Bot', 'Tuning Controls', self.crop_params['B_Bot'], 720, nothing)
        cv2.createTrackbar('B_Left', 'Tuning Controls', self.crop_params['B_Left'], 1280, nothing)
        cv2.createTrackbar('B_Right', 'Tuning Controls', self.crop_params['B_Right'], 1280, nothing)
        
        # Left (회전 후: 가로 720, 세로 1280)
        cv2.createTrackbar('L_Top', 'Tuning Controls', self.crop_params['L_Top'], 1280, nothing)
        cv2.createTrackbar('L_Bot', 'Tuning Controls', self.crop_params['L_Bot'], 1280, nothing)
        cv2.createTrackbar('L_Left', 'Tuning Controls', self.crop_params['L_Left'], 720, nothing)
        cv2.createTrackbar('L_Right', 'Tuning Controls', self.crop_params['L_Right'], 720, nothing)
        
        # Right (회전 후: 가로 720, 세로 1280)
        cv2.createTrackbar('R_Top', 'Tuning Controls', self.crop_params['R_Top'], 1280, nothing)
        cv2.createTrackbar('R_Bot', 'Tuning Controls', self.crop_params['R_Bot'], 1280, nothing)
        cv2.createTrackbar('R_Left', 'Tuning Controls', self.crop_params['R_Left'], 720, nothing)
        cv2.createTrackbar('R_Right', 'Tuning Controls', self.crop_params['R_Right'], 720, nothing)
        
        # --- Car Size ---
        cv2.createTrackbar('Car_W', 'Tuning Controls', self.car_params['Width'], 1000, nothing)
        cv2.createTrackbar('Car_L', 'Tuning Controls', self.car_params['Length'], 1000, nothing)

    def update_params_from_trackbars(self):
        # Undistort
        self.undistort_params['fx'] = cv2.getTrackbarPos('K_fx', 'Tuning Controls')
        self.undistort_params['fy'] = cv2.getTrackbarPos('K_fy', 'Tuning Controls')
        k1_val = cv2.getTrackbarPos('D_k1 (+100)', 'Tuning Controls')
        self.undistort_params['k1'] = (k1_val - 100) / 100.0

        # Crop
        self.crop_params['F_Top'] = cv2.getTrackbarPos('F_Top', 'Tuning Controls')
        self.crop_params['F_Bot'] = cv2.getTrackbarPos('F_Bot', 'Tuning Controls')
        self.crop_params['F_Left'] = cv2.getTrackbarPos('F_Left', 'Tuning Controls')
        self.crop_params['F_Right'] = cv2.getTrackbarPos('F_Right', 'Tuning Controls')

        self.crop_params['B_Top'] = cv2.getTrackbarPos('B_Top', 'Tuning Controls')
        self.crop_params['B_Bot'] = cv2.getTrackbarPos('B_Bot', 'Tuning Controls')
        self.crop_params['B_Left'] = cv2.getTrackbarPos('B_Left', 'Tuning Controls')
        self.crop_params['B_Right'] = cv2.getTrackbarPos('B_Right', 'Tuning Controls')
        
        self.crop_params['L_Top'] = cv2.getTrackbarPos('L_Top', 'Tuning Controls')
        self.crop_params['L_Bot'] = cv2.getTrackbarPos('L_Bot', 'Tuning Controls')
        self.crop_params['L_Left'] = cv2.getTrackbarPos('L_Left', 'Tuning Controls')
        self.crop_params['L_Right'] = cv2.getTrackbarPos('L_Right', 'Tuning Controls')
        
        self.crop_params['R_Top'] = cv2.getTrackbarPos('R_Top', 'Tuning Controls')
        self.crop_params['R_Bot'] = cv2.getTrackbarPos('R_Bot', 'Tuning Controls')
        self.crop_params['R_Left'] = cv2.getTrackbarPos('R_Left', 'Tuning Controls')
        self.crop_params['R_Right'] = cv2.getTrackbarPos('R_Right', 'Tuning Controls')

        # Car
        self.car_params['Width'] = cv2.getTrackbarPos('Car_W', 'Tuning Controls')
        self.car_params['Length'] = cv2.getTrackbarPos('Car_L', 'Tuning Controls')

    def undistort_image(self, img):
        h, w = img.shape[:2]
        fx = self.undistort_params['fx']
        fy = self.undistort_params['fy']
        cx = w / 2
        cy = h / 2
        
        K = np.array([[fx, 0, cx],
                      [0, fy, cy],
                      [0,  0,  1]])
        
        D = np.array([self.undistort_params['k1'], 0, 0, 0])
        
        new_K, roi = cv2.getOptimalNewCameraMatrix(K, D, (w, h), 1, (w, h))
        undistorted_img = cv2.undistort(img, K, D, None, new_K)
        
        return undistorted_img

    def crop_image(self, image, t, b, l, r):
        if image is None: return None
        h, w = image.shape[:2]
        # 안전장치: 잘라내는 영역이 이미지보다 크면 원본 반환
        if t + b >= h or l + r >= w: return image
        return image[t : h - b, l : w - r]

    def sync_callback(self, front_msg, back_msg, left_msg, right_msg):
        try:
            self.update_params_from_trackbars()

            # 1. 변환
            front_cv = self.bridge.imgmsg_to_cv2(front_msg, 'bgr8')
            back_cv  = self.bridge.imgmsg_to_cv2(back_msg, 'bgr8')
            left_cv  = self.bridge.imgmsg_to_cv2(left_msg, 'bgr8')
            right_cv = self.bridge.imgmsg_to_cv2(right_msg, 'bgr8')

            # 2. 왜곡 보정
            front_ud = self.undistort_image(front_cv)
            back_ud  = self.undistort_image(back_cv)
            left_ud  = self.undistort_image(left_cv)
            right_ud = self.undistort_image(right_cv)

            # 3. 회전
            # Front/Back: 1280x720 (Rotated 180) -> 1280x720
            front_rot = cv2.rotate(front_ud, cv2.ROTATE_180)
            back_rot  = cv2.rotate(back_ud,  cv2.ROTATE_180)
            # Left/Right: 1280x720 (Rotated 90) -> 720x1280
            left_rot  = cv2.rotate(left_ud,  cv2.ROTATE_90_COUNTERCLOCKWISE)
            right_rot = cv2.rotate(right_ud, cv2.ROTATE_90_COUNTERCLOCKWISE) 

            # 4. 크롭 (모든 방향 적용)
            p = self.crop_params
            front_roi = self.crop_image(front_rot, p['F_Top'], p['F_Bot'], p['F_Left'], p['F_Right'])
            back_roi  = self.crop_image(back_rot,  p['B_Top'], p['B_Bot'], p['B_Left'], p['B_Right'])
            left_roi  = self.crop_image(left_rot,  p['L_Top'], p['L_Bot'], p['L_Left'], p['L_Right'])
            right_roi = self.crop_image(right_rot, p['R_Top'], p['R_Bot'], p['R_Left'], p['R_Right'])

            # 5. 스티칭
            Hf, Wf = front_roi.shape[:2]
            Hb, Wb = back_roi.shape[:2]
            Hl, Wl = left_roi.shape[:2]
            Hr, Wr = right_roi.shape[:2]

            car_w = self.car_params['Width']
            car_l = self.car_params['Length']

            mid_height = max(Hl, Hr, car_l)
            canvas_w = max(Wf, Wb, Wl + car_w + Wr)
            canvas_h = Hf + mid_height + Hb
            
            canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

            # 배치
            x_offset_f = (canvas_w - Wf) // 2
            canvas[0:Hf, x_offset_f : x_offset_f + Wf] = front_roi
            
            y_offset_b = Hf + mid_height
            x_offset_b = (canvas_w - Wb) // 2
            canvas[y_offset_b : y_offset_b + Hb, x_offset_b : x_offset_b + Wb] = back_roi
            
            y_offset_mid_l = Hf + (mid_height - Hl) // 2
            canvas[y_offset_mid_l : y_offset_mid_l + Hl, 0 : Wl] = left_roi
            
            y_offset_mid_r = Hf + (mid_height - Hr) // 2
            canvas[y_offset_mid_r : y_offset_mid_r + Hr, canvas_w - Wr : canvas_w] = right_roi

            # 6. 결과 발행
            final_msg = self.bridge.cv2_to_imgmsg(canvas, "bgr8")
            final_msg.header = front_msg.header
            self.publisher_.publish(final_msg)

            cv2.imshow("Surround View Tuning", canvas)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                print("\n" + "="*50)
                print("★ 현재 튜닝값 (복사해서 코드 __init__에 붙여넣으세요) ★")
                print("-" * 50)
                print(f"self.undistort_params = {{'fx': {self.undistort_params['fx']}, 'fy': {self.undistort_params['fy']}, 'cx': 640, 'cy': 360, 'k1': {self.undistort_params['k1']:.2f}, 'k2': 0, 'p1': 0, 'p2': 0}}")
                print(f"self.CROP_FRONT = {{'top': {p['F_Top']}, 'bottom': {p['F_Bot']}, 'left': {p['F_Left']}, 'right': {p['F_Right']}}}")
                print(f"self.CROP_BACK  = {{'top': {p['B_Top']}, 'bottom': {p['B_Bot']}, 'left': {p['B_Left']}, 'right': {p['B_Right']}}}")
                print(f"self.CROP_LEFT  = {{'top': {p['L_Top']}, 'bottom': {p['L_Bot']}, 'left': {p['L_Left']}, 'right': {p['L_Right']}}}")
                print(f"self.CROP_RIGHT = {{'top': {p['R_Top']}, 'bottom': {p['R_Bot']}, 'left': {p['R_Left']}, 'right': {p['R_Right']}}}")
                print(f"\nself.CAR_WIDTH_PX = {car_w}")
                print(f"self.CAR_LENGTH_PX = {car_l}")
                print("="*50 + "\n")

        except Exception as e:
            self.get_logger().error(f'Error: {e}')

def main(args=None):
    rclpy.init(args=args)
    node = SurroundViewNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()