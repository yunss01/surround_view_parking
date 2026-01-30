import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import yaml
import message_filters

# === 설정 ===
BEV_WIDTH = 850
BEV_HEIGHT = 1150

class StitchingNode(Node):
    def __init__(self):
        super().__init__('stitching_node')
        self.bridge = CvBridge()

        # 1. Intrinsics (K, D) 로드
        self.intrinsics = {}
        try:
            with open('fisheye_intrinsics.yaml', 'r') as f:
                data = yaml.safe_load(f)
                for name in ['front', 'back', 'left', 'right']:
                    self.intrinsics[name] = {
                        'K': np.array(data[name]['K']),
                        'D': np.array(data[name]['D']),
                        'NK': None, 'map1': None, 'map2': None
                    }
        except: pass # 에러 처리

        # 2. Extrinsics (H) 로드
        try:
            with open('calibration.yaml', 'r') as f:
                data = yaml.safe_load(f)
                self.H = {
                    'front': np.array(data['front']),
                    'back':  np.array(data['back']),
                    'left':  np.array(data['left']),
                    'right': np.array(data['right'])
                }
        except: pass

        # 3. 토픽 구독 (동기화)
        self.sub_f = message_filters.Subscriber(self, Image, '/fo_camera/image_raw')
        self.sub_b = message_filters.Subscriber(self, Image, '/rear_camera/image_raw')
        self.sub_l = message_filters.Subscriber(self, Image, '/left_camera/image_raw')
        self.sub_r = message_filters.Subscriber(self, Image, '/right_camera/image_raw')

        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.sub_f, self.sub_b, self.sub_l, self.sub_r], queue_size=10, slop=0.2
        )
        self.ts.registerCallback(self.callback)

        self.pub = self.create_publisher(Image, '/surround_view/image_raw', 10)

        # 4. 차량 이미지 (마스크용, 검은색 사각형)
        self.car_mask = np.zeros((BEV_HEIGHT, BEV_WIDTH, 3), dtype=np.uint8)
        cx, cy = BEV_WIDTH // 2, BEV_HEIGHT // 2
        # 차량 크기만큼 중앙을 가림 (검은색)
        cv2.rectangle(self.car_mask, (cx - 100, cy - 200), (cx + 100, cy + 200), (0, 0, 0), -1)

    def undistort_image(self, img, name):
        h, w = img.shape[:2]
        # 맵 생성 (최초 1회만 계산하여 캐싱 -> 성능 최적화)
        if self.intrinsics[name]['map1'] is None:
            K = self.intrinsics[name]['K']
            D = self.intrinsics[name]['D']
            NK = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K, D, (w, h), np.eye(3), balance=1.0)
            map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), NK, (w, h), cv2.CV_16SC2)
            self.intrinsics[name]['map1'] = map1
            self.intrinsics[name]['map2'] = map2
        
        return cv2.remap(img, self.intrinsics[name]['map1'], self.intrinsics[name]['map2'], interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    def callback(self, msg_f, msg_b, msg_l, msg_r):
        # 1. 변환
        img_f = self.undistort_image(self.bridge.imgmsg_to_cv2(msg_f, "bgr8"), 'front')
        img_b = self.undistort_image(self.bridge.imgmsg_to_cv2(msg_b, "bgr8"), 'back')
        img_l = self.undistort_image(self.bridge.imgmsg_to_cv2(msg_l, "bgr8"), 'left')
        img_r = self.undistort_image(self.bridge.imgmsg_to_cv2(msg_r, "bgr8"), 'right')

        # 2. IPM (Warp Perspective)
        bev_f = cv2.warpPerspective(img_f, self.H['front'], (BEV_WIDTH, BEV_HEIGHT))
        bev_b = cv2.warpPerspective(img_b, self.H['back'],  (BEV_WIDTH, BEV_HEIGHT))
        bev_l = cv2.warpPerspective(img_l, self.H['left'],  (BEV_WIDTH, BEV_HEIGHT))
        bev_r = cv2.warpPerspective(img_r, self.H['right'], (BEV_WIDTH, BEV_HEIGHT))

        # 3. 합성 (Stitching)
        # 단순하게는 덮어쓰기지만, 여기서는 검은색이 아닌 부분만 합칩니다.
        final_bev = np.zeros((BEV_HEIGHT, BEV_WIDTH, 3), dtype=np.uint8)

        # 우선순위: 뒤 -> 좌/우 -> 앞 (덮어쓰는 순서)
        final_bev = self.overlay_image(final_bev, bev_b)
        final_bev = self.overlay_image(final_bev, bev_l)
        final_bev = self.overlay_image(final_bev, bev_r)
        final_bev = self.overlay_image(final_bev, bev_f)

        # 4. 차량 위치 검은색 처리 (또는 차량 이미지 삽입)
        # 중앙에 검은 박스 그리기
        cx, cy = BEV_WIDTH // 2, BEV_HEIGHT // 2
        cv2.rectangle(final_bev, (cx - 75, cy - 150), (cx + 75, cy + 150), (0, 0, 0), -1)

        # 5. 결과 발행
        out_msg = self.bridge.cv2_to_imgmsg(final_bev, "bgr8")
        out_msg.header = msg_f.header
        self.pub.publish(out_msg)

        # cv2.imshow("Surround View", final_bev)
        # cv2.waitKey(1)

    def overlay_image(self, background, overlay):
        # overlay 이미지가 검은색(0)이 아닌 부분만 background에 복사
        gray = cv2.cvtColor(overlay, cv2.COLOR_BGR2GRAY)
        mask = gray > 0
        background[mask] = overlay[mask]
        return background

def main():
    rclpy.init()
    node = StitchingNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
