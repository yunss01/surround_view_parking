import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import yaml
import pickle
import os

# ==========================================
# 🛑 사용자 설정 (작성자님 환경에 맞춤)
# ==========================================
# 체커보드 교차점 개수 (작성자님은 10x7이 맞습니다)
CHESSBOARD_SIZE = (10, 7) 

# 해상도 (640x480 유지)
IMG_WIDTH = 640
IMG_HEIGHT = 480
# ==========================================

class IntrinsicCalibNode(Node):
    def __init__(self):
        super().__init__('intrinsic_calib_node')
        self.bridge = CvBridge()
        
        # 3D 기준 좌표 생성 (10x7 = 70개 포인트)
        self.objp = np.zeros((1, CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
        self.objp[0,:,:2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)

        # 데이터 저장소 초기화
        self.imgpoints = {'front':[], 'back':[], 'left':[], 'right':[]}
        self.img_shape = (IMG_WIDTH, IMG_HEIGHT)

        # 📂 [핵심] 기존 데이터 불러오기 (이어찍기 기능)
        if os.path.exists('capture_data.pkl'):
            try:
                with open('capture_data.pkl', 'rb') as f:
                    data = pickle.load(f)
                    # 기존 데이터가 비어있지 않은지 확인하며 병합
                    if 'imgpoints' in data:
                        for name in ['front', 'back', 'left', 'right']:
                            if name in data['imgpoints']:
                                self.imgpoints[name] = data['imgpoints'][name]
                        self.get_logger().info(f"📂 기존 데이터 로드 완료! (Front: {len(self.imgpoints['front'])}장, Left: {len(self.imgpoints['left'])}장...)")
                    else:
                        self.get_logger().warn("⚠️ pickle 파일 형식이 달라서 새로 시작합니다.")
            except Exception as e:
                self.get_logger().error(f"❌ 데이터 로드 실패: {e}")
        else:
            self.get_logger().info("✨ 기존 데이터 없음. 새로 시작합니다.")

        # 토픽 구독
        self.create_subscription(Image, '/fo_camera/image_raw', lambda m: self.callback(m, 'front'), 10)
        self.create_subscription(Image, '/rear_camera/image_raw',  lambda m: self.callback(m, 'back'), 10)
        self.create_subscription(Image, '/left_camera/image_raw',  lambda m: self.callback(m, 'left'), 10)
        self.create_subscription(Image, '/right_camera/image_raw', lambda m: self.callback(m, 'right'), 10)
        
        self.get_logger().info("📷 [Append Mode] Ready. 찍으면 기존 데이터 뒤에 추가됩니다!")

    def callback(self, msg, name):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # 카운트 표시
            cv2.putText(img, f"{name}: {len(self.imgpoints[name])}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow(f"Intrinsic {name}", img)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == 32: # Spacebar
                ret, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, 
                    cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE)
                
                if ret:
                    # 정밀화
                    corners2 = cv2.cornerSubPix(gray, corners, (3,3), (-1,-1), 
                                                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1))
                    
                    self.imgpoints[name].append(corners2)
                    self.get_logger().info(f"📸 {name} 추가됨! (누적: {len(self.imgpoints[name])}장)")
                    
                    cv2.drawChessboardCorners(img, CHESSBOARD_SIZE, corners2, ret)
                    cv2.imshow(f"Intrinsic {name}", img)
                    cv2.waitKey(100)
                else:
                    self.get_logger().warn(f"❌ {name}: 체커보드 못 찾음 (각도 조절 필요)")
            
            elif key == ord('s'):
                self.save_data_and_calibrate()
                
        except Exception as e:
            pass

    def save_data_and_calibrate(self):
        self.get_logger().info("💾 데이터를 합쳐서 저장 중...")
        
        # 1. 합친 데이터 저장 (덮어쓰기)
        try:
            with open('capture_data.pkl', 'wb') as f:
                pickle.dump({'imgpoints': self.imgpoints, 'img_shape': self.img_shape, 'objp': self.objp}, f)
            self.get_logger().info("✅ capture_data.pkl 업데이트 완료!")
        except Exception as e:
            self.get_logger().error(f"❌ 저장 실패: {e}")

        # 2. 캘리브레이션 시작
        calib_data = {}
        # 에러 방지용 플래그 (CHECK_COND 제거)
        flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_FIX_SKEW
        
        for name in ['front', 'back', 'left', 'right']:
            if len(self.imgpoints[name]) < 10:
                self.get_logger().warn(f"⚠️ {name} 데이터 부족 ({len(self.imgpoints[name])}). 건너뜀.")
                continue
                
            self.get_logger().info(f"⏳ {name} 계산 중... (데이터 {len(self.imgpoints[name])}장)")
            
            K = np.zeros((3, 3))
            D = np.zeros((4, 1))
            rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for _ in range(len(self.imgpoints[name]))]
            tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for _ in range(len(self.imgpoints[name]))]
            current_objpoints = [self.objp] * len(self.imgpoints[name])
            
            try:
                rms, _, _, _, _ = cv2.fisheye.calibrate(
                    current_objpoints, 
                    self.imgpoints[name], 
                    (IMG_WIDTH, IMG_HEIGHT), 
                    K, 
                    D, 
                    rvecs, 
                    tvecs, 
                    flags,
                    (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
                )
                calib_data[name] = {'K': K.tolist(), 'D': D.tolist()}
                self.get_logger().info(f"✨ {name} 성공! RMS: {rms:.4f}")
                
            except Exception as e:
                self.get_logger().error(f"❌ {name} 계산 실패: {e}")

        # 3. YAML 저장
        if calib_data:
            with open('fisheye_intrinsics.yaml', 'w') as f:
                yaml.dump(calib_data, f)
            self.get_logger().info("🎉 640x480용 fisheye_intrinsics.yaml 생성 완료!")
        
        rclpy.shutdown()

def main():
    rclpy.init()
    node = IntrinsicCalibNode()
    try: rclpy.spin(node)
    except: pass

if __name__ == '__main__':
    main()