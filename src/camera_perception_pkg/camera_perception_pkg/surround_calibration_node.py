import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import yaml
import os
import message_filters

# === [설정] ===
BEV_WIDTH = 850
BEV_HEIGHT = 1150

# 초기값 (대략적인 값 - 실행 후 슬라이더로 조절하세요)
INIT_PARAMS = {
    'front': {'flat': 200, 'zoom': 100, 'x': 425, 'y': 300},
    'back':  {'flat': 200, 'zoom': 100, 'x': 425, 'y': 850},
    'left':  {'flat': 200, 'zoom': 100, 'x': 250, 'y': 575},
    'right': {'flat': 200, 'zoom': 100, 'x': 600, 'y': 575}
}

class ManualTuner(Node):
    def __init__(self):
        super().__init__('manual_calibration_node')
        self.bridge = CvBridge()
        self.intrinsics = {}
        
        # 기본 왜곡 보정 파라미터 (이건 고정)
        self.balance = 0.0
        self.fov_scale = 0.6
        
        if not self.load_intrinsics():
            self.get_logger().error("🚨 fisheye_intrinsics.yaml 없음! 종료합니다.")
            return

        # 튜닝용 윈도우 생성
        cv2.namedWindow("Tuning Panel", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Tuning Panel", 600, 400)
        
        # 트랙바 생성
        cv2.createTrackbar("Select Cam (0:F,1:B,2:L,3:R)", "Tuning Panel", 0, 3, self.nothing)
        cv2.createTrackbar("Flatten (Perspective)", "Tuning Panel", 150, 500, self.nothing)
        cv2.createTrackbar("Zoom (Scale)", "Tuning Panel", 100, 300, self.nothing)
        cv2.createTrackbar("X Move", "Tuning Panel", 425, 850, self.nothing)
        cv2.createTrackbar("Y Move", "Tuning Panel", 575, 1150, self.nothing)

        # 현재 튜닝값 저장소 (초기값 적용)
        self.params = INIT_PARAMS.copy()
        self.prev_cam_idx = -1
        
        # 토픽 구독
        self.sub_f = message_filters.Subscriber(self, Image, '/fo_camera/image_raw')
        self.sub_b = message_filters.Subscriber(self, Image, '/rear_camera/image_raw')
        self.sub_l = message_filters.Subscriber(self, Image, '/left_camera/image_raw')
        self.sub_r = message_filters.Subscriber(self, Image, '/right_camera/image_raw')
        
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.sub_f, self.sub_b, self.sub_l, self.sub_r], queue_size=10, slop=0.2
        )
        self.ts.registerCallback(self.callback)
        
        self.get_logger().info("🚀 수동 튜닝 모드 시작! 'S'를 눌러 저장하세요.")

    def nothing(self, x): pass

    def load_intrinsics(self):
        try:
            with open('fisheye_intrinsics.yaml', 'r') as f:
                data = yaml.safe_load(f)
                for name in ['front', 'back', 'left', 'right']:
                    self.intrinsics[name] = {
                        'K': np.array(data[name]['K']), 'D': np.array(data[name]['D']),
                        'map1': None, 'map2': None
                    }
            return True
        except: return False

    def undistort(self, img, name):
        h, w = img.shape[:2]
        if self.intrinsics[name]['map1'] is None:
            K = self.intrinsics[name]['K']
            D = self.intrinsics[name]['D']
            NK = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                K, D, (w, h), np.eye(3), balance=self.balance, fov_scale=self.fov_scale
            )
            map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), NK, (w, h), cv2.CV_16SC2)
            self.intrinsics[name]['map1'] = map1
            self.intrinsics[name]['map2'] = map2
        return cv2.remap(img, self.intrinsics[name]['map1'], self.intrinsics[name]['map2'], interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    def get_homography(self, name, p):
        # 슬라이더 값으로 Homography 행렬 생성
        # 간단한 원리: 사다리꼴(Trapizoid) -> 직사각형(Rect) 변환
        
        flat_val = p['flat']  # 원근감 보정 강도 (클수록 많이 폄)
        zoom_val = p['zoom'] / 100.0
        dx = p['x'] - (BEV_WIDTH / 2) # 중심 기준 이동량
        dy = p['y'] - (BEV_HEIGHT / 2)
        
        w, h = 640, 480 # 입력 이미지 크기 (undistorted 기준)
        
        # 소스 좌표 (사다리꼴) - 위쪽이 좁은 형태
        src_pts = np.float32([
            [0 + flat_val, 0],         # Top-Left
            [w - flat_val, 0],         # Top-Right
            [w, h],                    # Bottom-Right
            [0, h]                     # Bottom-Left
        ])
        
        # 목적지 좌표 (직사각형 + 줌 + 이동)
        bev_cx, bev_cy = BEV_WIDTH / 2, BEV_HEIGHT / 2
        
        # 기본적으로 이미지를 중앙에 놓고 줌/이동 적용
        dst_w = w * zoom_val
        dst_h = h * zoom_val
        
        dst_pts = np.float32([
            [bev_cx - dst_w/2 + dx, bev_cy - dst_h/2 + dy],
            [bev_cx + dst_w/2 + dx, bev_cy - dst_h/2 + dy],
            [bev_cx + dst_w/2 + dx, bev_cy + dst_h/2 + dy],
            [bev_cx - dst_w/2 + dx, bev_cy + dst_h/2 + dy]
        ])
        
        # [핵심] 카메라 방향에 따라 회전 (Front는 그대로, 나머지는 회전)
        # 이미지를 회전시키는 게 아니라, '좌표 매핑'을 회전시킴
        if name == 'back': 
            # 180도 회전 (Top-Left가 Bottom-Right로 가도록 점 순서 변경)
            dst_pts = np.roll(dst_pts, 2, axis=0) 
        elif name == 'left': 
            # 반시계 90도 (Top-Left가 Bottom-Left로 가도록)
             dst_pts = np.roll(dst_pts, 1, axis=0)
        elif name == 'right': 
            # 시계 90도 (Top-Left가 Top-Right로 가도록)
             dst_pts = np.roll(dst_pts, -1, axis=0)

        H = cv2.getPerspectiveTransform(src_pts, dst_pts)
        return H

    def callback(self, f, b, l, r):
        # 1. 현재 선택된 카메라 확인 및 슬라이더 값 읽기
        cam_idx = cv2.getTrackbarPos("Select Cam (0:F,1:B,2:L,3:R)", "Tuning Panel")
        cam_names = ['front', 'back', 'left', 'right']
        curr_cam = cam_names[cam_idx]
        
        # 2. 슬라이더 값 업데이트 (카메라 변경 시 값 로드 / 유지 시 값 저장)
        if self.prev_cam_idx != cam_idx:
            # 카메라가 바뀌었다! -> 저장된 값을 슬라이더에 반영
            p = self.params[curr_cam]
            cv2.setTrackbarPos("Flatten (Perspective)", "Tuning Panel", p['flat'])
            cv2.setTrackbarPos("Zoom (Scale)", "Tuning Panel", p['zoom'])
            cv2.setTrackbarPos("X Move", "Tuning Panel", p['x'])
            cv2.setTrackbarPos("Y Move", "Tuning Panel", p['y'])
            self.prev_cam_idx = cam_idx
        else:
            # 같은 카메라 -> 슬라이더 값을 변수에 저장
            self.params[curr_cam]['flat'] = cv2.getTrackbarPos("Flatten (Perspective)", "Tuning Panel")
            self.params[curr_cam]['zoom'] = cv2.getTrackbarPos("Zoom (Scale)", "Tuning Panel")
            self.params[curr_cam]['x'] = cv2.getTrackbarPos("X Move", "Tuning Panel")
            self.params[curr_cam]['y'] = cv2.getTrackbarPos("Y Move", "Tuning Panel")

        # 3. 이미지 처리 및 합성
        imgs = {
            'front': self.bridge.imgmsg_to_cv2(f, "bgr8"),
            'back': self.bridge.imgmsg_to_cv2(b, "bgr8"),
            'left': self.bridge.imgmsg_to_cv2(l, "bgr8"),
            'right': self.bridge.imgmsg_to_cv2(r, "bgr8")
        }
        
        final_bev = np.zeros((BEV_HEIGHT, BEV_WIDTH, 3), dtype=np.uint8)
        
        # 4개 카메라 모두 변환 후 합성
        for name in cam_names:
            undist = self.undistort(imgs[name], name)
            H = self.get_homography(name, self.params[name])
            warped = cv2.warpPerspective(undist, H, (BEV_WIDTH, BEV_HEIGHT))
            
            # 합성 (단순 덮어쓰기)
            mask = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY) > 10
            final_bev[mask] = warped[mask]
            
            # [선택된 카메라 강조] 빨간 테두리 표시 (미리보기용)
            if name == curr_cam:
                # 선택된 카메라의 변환 결과만 따로 작게 보여줌
                preview_small = cv2.resize(warped, (300, 400))
                cv2.imshow("Selected Cam Preview", preview_small)

        # 4. 결과 출력
        # 차량 박스 (중앙)
        cx, cy = BEV_WIDTH//2, BEV_HEIGHT//2
        cv2.rectangle(final_bev, (cx-60, cy-120), (cx+60, cy+120), (0,0,0), -1) # 차량
        
        # 보기 좋게 축소해서 보여줌
        preview_main = cv2.resize(final_bev, (BEV_WIDTH//2, BEV_HEIGHT//2))
        cv2.imshow("Main Stitching View", preview_main)
        
        # 5. 키 입력 처리 (저장)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            self.save_calibration()

    def save_calibration(self):
        # 현재 튜닝된 H 행렬들을 계산해서 저장
        H_dict = {}
        for name in ['front', 'back', 'left', 'right']:
            H_dict[name] = self.get_homography(name, self.params[name]).tolist()
            
        save_data = {
            'homography': H_dict,
            'settings': {'fov_scale': self.fov_scale, 'balance': self.balance}
        }
        
        with open('calibration.yaml', 'w') as f:
            yaml.dump(save_data, f)
        self.get_logger().info("💾 calibration.yaml 저장 완료! 이제 Stitching 노드를 켜보세요.")

def main():
    rclpy.init()
    node = ManualTuner()
    try: rclpy.spin(node)
    except: pass
    cv2.destroyAllWindows()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()