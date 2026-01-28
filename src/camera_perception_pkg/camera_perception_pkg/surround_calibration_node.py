import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import yaml
import os

# ==============================================================================
# 📏 [고정 설정] parking_spot_extractor_node.py와 동일한 스케일 적용
# ==============================================================================
CAR_LENGTH_M = 1.0      # 차량 실제 길이 (m)
CAR_LENGTH_PX = 340.0    # 화면상 차량 길이 (pixels)

# 1미터가 몇 픽셀인지 자동 계산 (약 118.88 px)
PIXELS_PER_METER = CAR_LENGTH_PX / CAR_LENGTH_M 

# 체커보드 정보 (10x7 보드 -> 내부 코너 9x6)
CHESSBOARD_COLS = 9       
CHESSBOARD_ROWS = 6       
SQUARE_SIZE_M = 0.025     # 25mm

# 결과 이미지 크기 (기존 코드와 동일하게 유지)
BEV_W = 850
BEV_H = 1150
CENTER_X = BEV_W // 2
CENTER_Y = BEV_H // 2

# ==============================================================================
# ✏️ [사용자 입력 필수] 줄자로 잰 거리 (미터 단위)만 수정하세요!
# ==============================================================================
OFFSETS = {
    # 예시: 차 중심에서 앞/뒤/좌/우 보드까지 1.0m 떨어져 있음
    'front_dist': 1.0,  
    'back_dist':  1.0,  
    'left_dist':  1.0,  
    'right_dist': 1.0   
}
# ==============================================================================

class CalibrationNode(Node):
    def __init__(self):
        super().__init__('calibration_node')
        self.bridge = CvBridge()
        self.homography_matrices = {'front': None, 'back': None, 'left': None, 'right': None}

        self.create_subscription(Image, '/camera/front', lambda m: self.img_callback(m, 'front'), 10)
        self.create_subscription(Image, '/camera/back',  lambda m: self.img_callback(m, 'back'), 10)
        self.create_subscription(Image, '/camera/left',  lambda m: self.img_callback(m, 'left'), 10)
        self.create_subscription(Image, '/camera/right', lambda m: self.img_callback(m, 'right'), 10)

        self.current_imgs = {}
        self.get_logger().info(f"📷 Calibration Ready. Scale: {PIXELS_PER_METER:.2f} px/m")

    def img_callback(self, msg, name):
        self.current_imgs[name] = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        cv2.imshow(f"Cam: {name}", self.current_imgs[name])
        self.check_key()

    def find_homography(self, name):
        if name not in self.current_imgs: return
        img = self.current_imgs[name]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (CHESSBOARD_COLS, CHESSBOARD_ROWS), None)

        if ret:
            # 1. 이미지 상의 좌표
            src_pts = corners.reshape(-1, 2).astype(np.float32)

            # 2. BEV 상의 목표 좌표 (자동 계산)
            dst_pts = self.calc_bev_points(name)

            # 3. 행렬 계산
            H, _ = cv2.findHomography(src_pts, dst_pts)
            self.homography_matrices[name] = H.tolist()
            
            # 검증: 미리보기
            warped = cv2.warpPerspective(img, H, (BEV_W, BEV_H))
            cv2.imshow("Warped Preview", warped)
            self.get_logger().info(f"✅ {name} Calibrated! (Distance: {OFFSETS[name+'_dist']}m)")
        else:
            self.get_logger().warn(f"❌ {name}: 체커보드 못 찾음. 각도를 조절하거나 보드를 평평한 곳에 두세요.")

    def calc_bev_points(self, name):
        obj_pts = []
        
        # 보드 전체 크기 (픽셀)
        board_w_px = (CHESSBOARD_COLS - 1) * SQUARE_SIZE_M * PIXELS_PER_METER
        board_h_px = (CHESSBOARD_ROWS - 1) * SQUARE_SIZE_M * PIXELS_PER_METER
        
        # 기준점 (보드 좌상단) 계산
        start_x, start_y = 0, 0
        dist_px = OFFSETS[f'{name}_dist'] * PIXELS_PER_METER # 미터 -> 픽셀 변환
        
        if name == 'front':
            # 차 중심보다 위쪽(-y), 가로 정렬
            start_x = CENTER_X - (board_w_px / 2)
            start_y = CENTER_Y - dist_px
            
        elif name == 'back':
            # 차 중심보다 아래쪽(+y), 가로 정렬
            start_x = CENTER_X - (board_w_px / 2)
            start_y = CENTER_Y + dist_px
            
        elif name == 'left':
            # 차 중심보다 왼쪽(-x), 세로 정렬이라 가정 (보드를 차랑 나란히 놓았을 때)
            start_x = CENTER_X - dist_px
            start_y = CENTER_Y - (board_h_px / 2) 
            
        elif name == 'right':
            # 차 중심보다 오른쪽(+x), 세로 정렬
            start_x = CENTER_X + dist_px
            start_y = CENTER_Y - (board_h_px / 2)

        # 좌표 생성
        for i in range(CHESSBOARD_ROWS):
            for j in range(CHESSBOARD_COLS):
                if name in ['front', 'back']:
                    # 가로 배치 (j가 x축, i가 y축)
                    px = start_x + j * (SQUARE_SIZE_M * PIXELS_PER_METER)
                    py = start_y + i * (SQUARE_SIZE_M * PIXELS_PER_METER)
                else:
                    # 세로 배치 (i가 x축, j가 y축 - 보드가 90도 돌았다고 가정)
                    # *중요: 실제 보드 놓는 방향에 따라 여기가 다를 수 있음. 
                    # 일단 차 옆에 '나란히(길게)' 놓았다고 가정함.
                    px = start_x + i * (SQUARE_SIZE_M * PIXELS_PER_METER)
                    py = start_y + j * (SQUARE_SIZE_M * PIXELS_PER_METER)
                obj_pts.append([px, py])
                
        return np.array(obj_pts, dtype=np.float32)

    def check_key(self):
        key = cv2.waitKey(1) & 0xFF
        if key == ord('f'): self.find_homography('front')
        elif key == ord('b'): self.find_homography('back')
        elif key == ord('l'): self.find_homography('left')
        elif key == ord('r'): self.find_homography('right')
        elif key == ord('s'): 
            with open('calibration.yaml', 'w') as f:
                yaml.dump(self.homography_matrices, f)
            self.get_logger().info("💾 calibration.yaml 저장 완료! 종료합니다.")
            rclpy.shutdown()

def main():
    rclpy.init()
    node = CalibrationNode()
    try: rclpy.spin(node)
    except: pass

if __name__ == '__main__':
    main()
