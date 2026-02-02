import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import pickle
import yaml

# === 설정 ===
CHESSBOARD_SIZE = (10, 7) # 내부 코너점 개수
img_shape = None

class IntrinsicCalibNode(Node):
    def __init__(self):
        super().__init__('intrinsic_calib_node')
        self.bridge = CvBridge()
        
        self.imgpoints = {'front':[], 'back':[], 'left':[], 'right':[]}
        
        # 3D 기준 좌표 생성
        self.objp = np.zeros((1, CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
        self.objp[0,:,:2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)

        # 서브픽셀 정밀화 조건 (정확도 향상 -> 캘리브레이션 성공률 UP)
        self.subpix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)

        # 토픽 구독
        self.create_subscription(Image, '/fo_camera/image_raw', lambda m: self.callback(m, 'front'), 10)
        self.create_subscription(Image, '/rear_camera/image_raw',  lambda m: self.callback(m, 'back'), 10)
        self.create_subscription(Image, '/left_camera/image_raw',  lambda m: self.callback(m, 'left'), 10)
        self.create_subscription(Image, '/right_camera/image_raw', lambda m: self.callback(m, 'right'), 10)
        
        self.get_logger().info("📷 [Safety Mode] Ready. Capture images with [Space], Save with [s].")

    def callback(self, msg, name):
        global img_shape
        try:
            img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            if img_shape is None: img_shape = img.shape[:2][::-1]
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
                    # [핵심] 서브픽셀 정밀화 (이게 없으면 계산하다 죽을 확률 높음)
                    corners2 = cv2.cornerSubPix(gray, corners, (3,3), (-1,-1), self.subpix_criteria)
                    
                    self.imgpoints[name].append(corners2)
                    self.get_logger().info(f"📸 {name} Captured! (Total: {len(self.imgpoints[name])})")
                    
                    cv2.drawChessboardCorners(img, CHESSBOARD_SIZE, corners2, ret)
                    cv2.imshow(f"Intrinsic {name}", img)
                    cv2.waitKey(100) # 잠깐 멈춤
                else:
                    self.get_logger().warn(f"❌ {name}: Board not found!")
            
            elif key == ord('s'):
                self.save_data_and_calibrate()
                
        except Exception as e:
            pass

    def save_data_and_calibrate(self):
        self.get_logger().info("💾 Saving raw data to 'capture_data.pkl' first...")
        
        # 1. 데이터 우선 백업 (계산하다 죽어도 살릴 수 있게)
        try:
            with open('capture_data.pkl', 'wb') as f:
                pickle.dump({'imgpoints': self.imgpoints, 'img_shape': img_shape, 'objp': self.objp}, f)
            self.get_logger().info("✅ Data Backup Complete! Now starting calculation...")
        except Exception as e:
            self.get_logger().error(f"❌ Backup failed: {e}")

        # 2. 캘리브레이션 시작
        calib_data = {}
        flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_FIX_SKEW
        
        for name in ['front', 'back', 'left', 'right']:
            if len(self.imgpoints[name]) < 10:
                self.get_logger().warn(f"⚠️ {name} has insufficient data ({len(self.imgpoints[name])}). Skipping.")
                continue
                
            self.get_logger().info(f"⏳ Processing {name}...")
            
            K = np.zeros((3, 3))
            D = np.zeros((4, 1))
            rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for _ in range(len(self.imgpoints[name]))]
            tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for _ in range(len(self.imgpoints[name]))]
            
            current_objpoints = [self.objp] * len(self.imgpoints[name])
            
            try:
                # 캘리브레이션 실행 (예외 처리 감싸기)
                rms, _, _, _, _ = cv2.fisheye.calibrate(
                    current_objpoints, 
                    self.imgpoints[name], 
                    img_shape, 
                    K, 
                    D, 
                    rvecs, 
                    tvecs, 
                    flags,
                    (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
                )
                calib_data[name] = {'K': K.tolist(), 'D': D.tolist()}
                self.get_logger().info(f"✨ {name} Success! RMS: {rms:.4f}")
                
            except cv2.error as e:
                self.get_logger().error(f"❌ OpenCV Error in {name}: {e}")
            except Exception as e:
                self.get_logger().error(f"❌ Crash in {name}: {e}")

        # 3. 결과 저장
        if calib_data:
            with open('fisheye_intrinsics.yaml', 'w') as f:
                yaml.dump(calib_data, f)
            self.get_logger().info("🎉 fisheye_intrinsics.yaml Saved Successfully!")
        else:
            self.get_logger().error("😱 No cameras were calibrated successfully.")
            
        rclpy.shutdown()

def main():
    rclpy.init()
    node = IntrinsicCalibNode()
    try: rclpy.spin(node)
    except: pass

if __name__ == '__main__':
    main()