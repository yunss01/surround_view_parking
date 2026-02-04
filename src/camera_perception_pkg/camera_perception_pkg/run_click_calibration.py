import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import yaml
import sys

# ==========================================
# ⚙️ 설정
# ==========================================
# 체커보드를 BEV 상에서 몇 픽셀 크기의 정사각형으로 만들 것인가?
# (이 값이 클수록 이미지가 크게 줌인 됩니다)
SQUARE_SIZE = 120 

# 이미지 저장 경로
YAML_INTRINSICS = 'fisheye_intrinsics.yaml'
YAML_CALIBRATION = 'calibration.yaml'
# ==========================================

class ClickCalibrationNode(Node):
    def __init__(self):
        super().__init__('click_calibration_node')
        self.bridge = CvBridge()
        self.intrinsics = {}
        
        # 1. 렌즈 왜곡 정보 로드
        if not self.load_intrinsics():
            self.get_logger().error(f"🚨 {YAML_INTRINSICS} 파일이 없습니다! 캘리브레이션부터 하세요.")
            sys.exit(1)

        # 2. 작업 변수
        self.camera_order = ['front', 'back', 'left', 'right']
        self.current_idx = 0
        self.clicked_points = []
        self.homography_dict = {}
        self.latest_img = None

        # 3. 모든 카메라 구독 (하지만 현재 순서의 카메라만 처리함)
        self.create_subscription(Image, '/fo_camera/image_raw', lambda m: self.img_callback(m, 'front'), 10)
        self.create_subscription(Image, '/rear_camera/image_raw',  lambda m: self.img_callback(m, 'back'), 10)
        self.create_subscription(Image, '/left_camera/image_raw',  lambda m: self.img_callback(m, 'left'), 10)
        self.create_subscription(Image, '/right_camera/image_raw', lambda m: self.img_callback(m, 'right'), 10)

        self.get_logger().info("🚀 클릭 캘리브레이션 시작! [Front] 카메라의 체커보드 4개를 클릭하세요.")
        self.get_logger().info("순서: 좌상 -> 우상 -> 우하 -> 좌하 (시계 방향)")

        # GUI 타이머
        self.create_timer(0.1, self.gui_loop)

    def load_intrinsics(self):
        try:
            with open(YAML_INTRINSICS, 'r') as f:
                data = yaml.safe_load(f)
                for name in ['front', 'back', 'left', 'right']:
                    self.intrinsics[name] = {
                        'K': np.array(data[name]['K']), 'D': np.array(data[name]['D']),
                        'map1': None, 'map2': None
                    }
            return True
        except: return False

    def undistort(self, img, name):
        # 기존 코드의 undistort 로직 그대로 사용
        h, w = img.shape[:2]
        if self.intrinsics[name]['map1'] is None:
            K = self.intrinsics[name]['K']
            D = self.intrinsics[name]['D']
            # balance=0.0 (최대한 많이 보존), fov_scale 등 조절 가능
            NK = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                K, D, (w, h), np.eye(3), balance=0.0, fov_scale=1.0
            )
            map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), NK, (w, h), cv2.CV_16SC2)
            self.intrinsics[name]['map1'] = map1
            self.intrinsics[name]['map2'] = map2
        return cv2.remap(img, self.intrinsics[name]['map1'], self.intrinsics[name]['map2'], interpolation=cv2.INTER_LINEAR)

    def img_callback(self, msg, name):
        # 현재 작업 중인 카메라의 이미지만 저장
        target_cam = self.camera_order[self.current_idx]
        if name == target_cam:
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.latest_img = self.undistort(frame, name)

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.clicked_points) < 4:
                self.clicked_points.append((x, y))
                print(f"✅ 포인트 {len(self.clicked_points)}/4 : ({x}, {y})")

    def gui_loop(self):
        if self.latest_img is None: return
        
        target_cam = self.camera_order[self.current_idx]
        display = self.latest_img.copy()

        # 클릭한 점 그리기
        for pt in self.clicked_points:
            cv2.circle(display, pt, 5, (0, 0, 255), -1) # 빨간 점
        
        # 선 연결 (시각적 도움)
        if len(self.clicked_points) > 1:
            for i in range(len(self.clicked_points)-1):
                cv2.line(display, self.clicked_points[i], self.clicked_points[i+1], (0,255,0), 2)
        if len(self.clicked_points) == 4:
            cv2.line(display, self.clicked_points[3], self.clicked_points[0], (0,255,0), 2)

        win_name = f"Calibration: {target_cam.upper()} (Click 4 points)"
        cv2.imshow("Calib Window", display)
        cv2.setMouseCallback("Calib Window", self.mouse_callback)

        key = cv2.waitKey(1) & 0xFF

        # 4개 점이 다 찍히면 -> 'Space'를 눌러 변환 확인
        if len(self.clicked_points) == 4:
            # === [작성자님의 요청 코드 삽입 부분] ===
            src = np.float32(self.clicked_points)
            
            # 화면 중앙 좌표 계산
            h, w = display.shape[:2]
            cx, cy = w // 2, h // 2
            
            # 목표: 중앙에 정사각형 만들기 (시계 방향 순서: 좌상->우상->우하->좌하)
            dst = np.float32([
                [cx - SQUARE_SIZE, cy - SQUARE_SIZE], # 좌상
                [cx + SQUARE_SIZE, cy - SQUARE_SIZE], # 우상
                [cx + SQUARE_SIZE, cy + SQUARE_SIZE], # 우하
                [cx - SQUARE_SIZE, cy + SQUARE_SIZE]  # 좌하
            ])

            # 변환 행렬(H) 계산
            H = cv2.getPerspectiveTransform(src, dst)
            
            # 미리보기 보여주기
            warped = cv2.warpPerspective(self.latest_img, H, (w, h)) # 일단 원본 크기로 봅니다
            cv2.putText(warped, "Press 'Enter' to Save & Next, 'R' to Retry", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.imshow("Preview (BEV)", warped)
            
            if key == 13: # Enter 키
                self.homography_dict[target_cam] = H.tolist()
                print(f"💾 {target_cam} 설정 완료!")
                
                # 다음 카메라로 이동
                self.current_idx += 1
                self.clicked_points = [] # 점 초기화
                cv2.destroyWindow("Preview (BEV)")
                
                # 모든 카메라 완료 시
                if self.current_idx >= 4:
                    self.save_and_exit()
            
            elif key == ord('r'): # 재시도
                self.clicked_points = []
                print("🔄 다시 찍어주세요.")
                cv2.destroyWindow("Preview (BEV)")

    def save_and_exit(self):
        print("\n💾 모든 카메라 설정 완료! calibration.yaml 저장 중...")
        data = {
            'homography': self.homography_dict,
            'settings': {'fov_scale': 1.0, 'balance': 0.0}
        }
        with open(YAML_CALIBRATION, 'w') as f:
            yaml.dump(data, f)
        
        print(f"🎉 저장 완료: {YAML_CALIBRATION}")
        print("이제 'surround_stitching_node.py'를 실행하면 이미지가 평평하게 보일 겁니다.")
        print("위치 이동이 필요하면 'surround_calibration_node.py'를 켜서 X, Y만 움직이세요.")
        cv2.destroyAllWindows()
        self.destroy_node()
        rclpy.shutdown()
        sys.exit(0)

def main():
    rclpy.init()
    node = ClickCalibrationNode()
    try: rclpy.spin(node)
    except SystemExit: pass
    except: pass

if __name__ == '__main__':
    main()