import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import yaml

# === 설정 ===
CHESSBOARD_SIZE = (10, 7) # 내부 코너점 개수
img_shape = None

class IntrinsicCalibNode(Node):
    def __init__(self):
        super().__init__('intrinsic_calib_node')
        self.bridge = CvBridge()
        
        # 4개 카메라 데이터 각각 수집
        self.objpoints = [] # 3d point in real world space
        self.imgpoints = {'front':[], 'back':[], 'left':[], 'right':[]}
        
        # 기준 좌표 (z=0)
        self.objp = np.zeros((1, CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
        self.objp[0,:,:2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)

        self.create_subscription(Image, '/fo_camera/image_raw', lambda m: self.callback(m, 'front'), 10)
        self.create_subscription(Image, '/rear_camera/image_raw',  lambda m: self.callback(m, 'back'), 10)
        self.create_subscription(Image, '/left_camera/image_raw',  lambda m: self.callback(m, 'left'), 10)
        self.create_subscription(Image, '/right_camera/image_raw', lambda m: self.callback(m, 'right'), 10)

        self.get_logger().info("📷 Intrinsic Calibration: 체커보드를 다양한 각도에서 보여주고 [Space]를 눌러 캡처하세요. [s] 저장.")

    def callback(self, msg, name):
        global img_shape
        img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        if img_shape is None: img_shape = img.shape[:2][::-1]
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 화면 표시
        cv2.imshow(f"Intrinsic {name}", img)
        key = cv2.waitKey(1) & 0xFF
        
        if key == 32: # Spacebar
            ret, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, 
                cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE)
            if ret:
                self.imgpoints[name].append(corners)
                self.objpoints.append(self.objp) # objpoints는 공통
                self.get_logger().info(f"✅ {name} Captured! (Total: {len(self.imgpoints[name])})")
                
                # 캡처된 것 잠깐 보여주기
                cv2.drawChessboardCorners(img, CHESSBOARD_SIZE, corners, ret)
                cv2.imshow(f"Intrinsic {name}", img)
                cv2.waitKey(500)
        
        elif key == ord('s'):
            self.calibrate_and_save()

    def calibrate_and_save(self):
        self.get_logger().info("⏳ Calculating Fisheye Parameters... (This takes time)")
        calib_data = {}
        
        for name in ['front', 'back', 'left', 'right']:
            if len(self.imgpoints[name]) < 10:
                self.get_logger().warn(f"⚠️ {name} 데이터 부족! (최소 10장 권장)")
                continue
                
            # OpenCV Fisheye Calibration
            N_OK = len(self.imgpoints[name])
            K = np.zeros((3, 3))
            D = np.zeros((4, 1))
            rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
            tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
            
            # Use objpoints repeated for this camera count
            current_objpoints = [self.objp] * N_OK
            
            rms, _, _, _, _ = cv2.fisheye.calibrate(
                current_objpoints, 
                self.imgpoints[name], 
                img_shape, 
                K, 
                D, 
                rvecs, 
                tvecs, 
                cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_CHECK_COND + cv2.fisheye.CALIB_FIX_SKEW
            )
            
            calib_data[name] = {
                'K': K.tolist(),
                'D': D.tolist()
            }
            self.get_logger().info(f"✨ {name} Calibrated! RMS: {rms:.4f}")

        with open('fisheye_intrinsics.yaml', 'w') as f:
            yaml.dump(calib_data, f)
        self.get_logger().info("💾 fisheye_intrinsics.yaml Saved!")
        rclpy.shutdown()

def main():
    rclpy.init()
    node = IntrinsicCalibNode()
    try: rclpy.spin(node)
    except: pass

if __name__ == '__main__':
    main()
