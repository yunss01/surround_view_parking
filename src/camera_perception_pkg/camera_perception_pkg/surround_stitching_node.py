import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import yaml
import message_filters

BEV_WIDTH = 850
BEV_HEIGHT = 1150

class StitchingNode(Node):
    def __init__(self):
        super().__init__('stitching_node')
        self.bridge = CvBridge()
        
        self.balance = 0.0
        self.fov_scale = 0.6
        
        self.intrinsics = {}
        self.H = {}
        self.load_files() 

        self.sub_f = message_filters.Subscriber(self, Image, '/fo_camera/image_raw')
        self.sub_b = message_filters.Subscriber(self, Image, '/rear_camera/image_raw')
        self.sub_l = message_filters.Subscriber(self, Image, '/left_camera/image_raw')
        self.sub_r = message_filters.Subscriber(self, Image, '/right_camera/image_raw')

        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.sub_f, self.sub_b, self.sub_l, self.sub_r], queue_size=10, slop=0.2
        )
        self.ts.registerCallback(self.callback)
        self.pub = self.create_publisher(Image, '/surround_view/image_raw', 10)
        self.get_logger().info(f"✅ Stitching Ready! (Applied Zoom: {self.fov_scale})")

    def load_files(self):
        try:
            with open('fisheye_intrinsics.yaml', 'r') as f:
                data = yaml.safe_load(f)
                for name in ['front', 'back', 'left', 'right']:
                    self.intrinsics[name] = {
                        'K': np.array(data[name]['K']), 'D': np.array(data[name]['D']),
                        'map1': None, 'map2': None
                    }
        except: pass

        try:
            with open('calibration.yaml', 'r') as f:
                data = yaml.safe_load(f)
                if 'homography' in data:
                    for name in ['front', 'back', 'left', 'right']:
                        self.H[name] = np.array(data['homography'][name])
                if 'settings' in data:
                    self.fov_scale = data['settings'].get('fov_scale', 0.6)
                    self.balance = data['settings'].get('balance', 0.0)
        except Exception as e:
            self.get_logger().error(f"파일 로드 실패: {e}")

    def undistort_and_warp(self, img, name):
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
            
        undist = cv2.remap(img, self.intrinsics[name]['map1'], self.intrinsics[name]['map2'], 
                           interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        
        if name in self.H:
            # [핵심 수정] H 행렬이 너무 튀는 값이면 강제로 보정하지 않음 (그대로 적용)
            # 하지만 이미지가 화면 밖으로 나간다면 여기서 평행이동(Translation) 값을 수정해야 함.
            # 일단은 그대로 warp 해보고, 안 나오면 캘리브레이션을 다시 하는게 맞음.
            # 여기서는 사용자 요청대로 일단 그대로 둡니다.
            return cv2.warpPerspective(undist, self.H[name], (BEV_WIDTH, BEV_HEIGHT))
            
        return np.zeros((BEV_HEIGHT, BEV_WIDTH, 3), dtype=np.uint8)

    def callback(self, f, b, l, r):
        try:
            imgs = {
                'front': self.bridge.imgmsg_to_cv2(f, "bgr8"),
                'back': self.bridge.imgmsg_to_cv2(b, "bgr8"),
                'left': self.bridge.imgmsg_to_cv2(l, "bgr8"),
                'right': self.bridge.imgmsg_to_cv2(r, "bgr8")
            }
            bevs = {k: self.undistort_and_warp(v, k) for k, v in imgs.items()}
            
            final = np.zeros((BEV_HEIGHT, BEV_WIDTH, 3), dtype=np.uint8)
            
            # [중요] 합성 순서 조정 (옆면을 먼저 깔고 앞뒤를 덮음)
            final = self.overlay(final, bevs['left'])
            final = self.overlay(final, bevs['right'])
            final = self.overlay(final, bevs['back'])
            final = self.overlay(final, bevs['front'])
            
            cx, cy = BEV_WIDTH//2, BEV_HEIGHT//2
            cv2.rectangle(final, (cx-60, cy-120), (cx+60, cy+120), (0,0,0), -1)
            
            self.pub.publish(self.bridge.cv2_to_imgmsg(final, "bgr8"))
            
            # 화면에 띄우기
            view = cv2.resize(final, (BEV_WIDTH//2, BEV_HEIGHT//2))
            cv2.imshow("Surround View", view)
            cv2.waitKey(1)
        except: pass

    def overlay(self, bg, fg):
        gray = cv2.cvtColor(fg, cv2.COLOR_BGR2GRAY)
        mask = gray > 10
        bg[mask] = fg[mask]
        return bg

def main():
    rclpy.init()
    node = StitchingNode()
    try: rclpy.spin(node)
    except: pass
    cv2.destroyAllWindows()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()