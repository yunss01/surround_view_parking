import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import argparse
import sys
import time

# ==========================================
# 📷 카메라 파라미터 (fisheye_intrinsics.yaml 값)
# ==========================================

K = np.array(
    [[194.21690210166952, 0.0, 327.62284556458144],
     [0.0, 194.4195894680052, 241.26407471528236],
     [0.0, 0.0, 1.0]], 
    dtype=np.float64)
 
D = np.array(
    [[-0.04298150918229655],
     [0.0354564971633773],
     [-0.03052103616020201],
     [0.005428186522181235]], 
    dtype=np.float64)

RESOLUTION = (640, 480)
BEV_SIZE = (640, 480) # 출력될 BEV 이미지 크기

# ==========================================
# 🛠️ 유틸리티 클래스 (작성자님 코드 유지)
# ==========================================
class PointSelector:
    def __init__(self, image, window_title="Select 4 Points"):
        self.image = image.copy()
        self.orig_image = image.copy()
        self.window_title = window_title
        self.points = []

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(self.points) < 4:
            self.points.append((x, y))
            cv2.circle(self.image, (x, y), 5, (0, 255, 0), -1)
            cv2.imshow(self.window_title, self.image)

    def select_points(self):
        cv2.namedWindow(self.window_title)
        cv2.setMouseCallback(self.window_title, self.mouse_callback)
        print(f"[{self.window_title}] 점 4개를 클릭하세요: 좌상->우상->우하->좌하")
        
        while True:
            cv2.imshow(self.window_title, self.image)
            key = cv2.waitKey(1) & 0xFF
            if key == 13 and len(self.points) == 4: # Enter
                break
            if key == ord('r'): # Reset
                self.points = []
                self.image = self.orig_image.copy()
                print("초기화되었습니다. 다시 찍으세요.")
            if key == ord('q'): # Quit
                sys.exit(0)

        cv2.destroyAllWindows()
        return np.array(self.points, dtype=np.float32)

def get_combined_map(K, D, src_points, dst_points, bev_size, orig_size):
    # (작성자님의 수식 로직 그대로 사용)
    bev_w, bev_h = bev_size
    
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    M_inv = np.linalg.inv(M)
    
    u_bev, v_bev = np.meshgrid(np.arange(bev_w), np.arange(bev_h))
    bev_coords = np.vstack((u_bev.flatten(), v_bev.flatten(), np.ones(bev_w * bev_h)))
    undist_coords_homo = M_inv @ bev_coords
    
    Z_inv = 1.0 / undist_coords_homo[2, :]
    u_undist = undist_coords_homo[0, :] * Z_inv
    v_undist = undist_coords_homo[1, :] * Z_inv
    
    xc = (u_undist - K[0, 2]) / K[0, 0]
    yc = (v_undist - K[1, 2]) / K[1, 1]
    r = np.sqrt(xc**2 + yc**2)
    
    theta = np.arctan(r)
    theta_d = theta * (1 + D[0] * theta**2 + D[1] * theta**4 + D[2] * theta**6 + D[3] * theta**8)
    
    scale = np.divide(theta_d, r, out=np.zeros_like(r), where=r!=0)
    x_dist = xc * scale
    y_dist = yc * scale
    
    map1 = (K[0, 0] * x_dist + K[0, 2]).reshape(bev_h, bev_w).astype(np.float32)
    map2 = (K[1, 1] * y_dist + K[1, 2]).reshape(bev_h, bev_w).astype(np.float32)
    
    return map1, map2

# ==========================================
# 🤖 ROS 2 노드 (이미지 1장 캡처용)
# ==========================================
class MapGeneratorNode(Node):
    def __init__(self, target_camera, topic_name):
        super().__init__('map_generator_node')
        self.target_camera = target_camera
        self.bridge = CvBridge()
        self.captured_image = None
        
        # 지정된 토픽 구독
        self.sub = self.create_subscription(
            Image, topic_name, self.listener_callback, 10
        )
        self.get_logger().info(f"📡 '{target_camera}' 카메라 연결 대기 중... (토픽: {topic_name})")

    def listener_callback(self, msg):
        if self.captured_image is None:
            self.captured_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.get_logger().info("📸 이미지 캡처 성공! 캘리브레이션을 시작합니다.")

# ==========================================
# 🚀 메인 실행 함수
# ==========================================
def main():
    # 1. 인자 파싱 (어떤 카메라를 작업할 것인가?)
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--camera", required=True, help="카메라 이름 (front, back, left, right)")
    parser.add_argument("-t", "--topic", required=True, help="ROS 토픽 이름 (예: /fo_camera/image_raw)")
    args = parser.parse_args()

    rclpy.init()
    node = MapGeneratorNode(args.camera, args.topic)

    # 2. 이미지 1장이 들어올 때까지 대기
    while rclpy.ok() and node.captured_image is None:
        rclpy.spin_once(node)

    if node.captured_image is None:
        print("❌ 이미지를 받지 못하고 종료합니다.")
        return

    # 3. 캡처된 이미지 가져오기
    frame = node.captured_image
    
    # 4. 카메라 위치별 목표 좌표(DST) 설정
    # 중요: 앞/뒤는 가로로 긴 직사각형, 좌/우는 세로로 긴 직사각형 형태로 펴야 이쁩니다.
    # (BEV 이미지 크기가 640x480일 때의 좌표 예시)
    
    # 이미지 중앙 좌표 (640x480 기준)
    cx, cy = 320, 240
    
    # 한 칸당 20픽셀로 가정
    # 10칸(긴쪽) = 200px
    # 7칸(짧은쪽) = 140px

    if args.camera in ['front', 'back']:
        # [앞/뒤 카메라] 
        # 체커보드 긴 쪽(10칸)을 가로로 두세요.
        # 가로 폭: 200px, 세로 높이: 140px
        half_w = 30 
        half_h = 21
        
        dst_points = np.array([
            [cx - half_w, cy - half_h],  # 좌상
            [cx + half_w, cy - half_h],  # 우상
            [cx + half_w, cy + half_h],  # 우하
            [cx - half_w, cy + half_h]   # 좌하
        ], dtype=np.float32)
    
    else:
        # [좌/우 카메라]
        # 체커보드 짧은 쪽(7칸)을 가로로 두세요. (차 밖으로 길게 뻗어나가는 모양)
        # 가로 폭: 140px, 세로 높이: 200px
        half_w = 21
        half_h = 30
        
        dst_points = np.array([
            [cx - half_w, cy - half_h],  # 좌상
            [cx + half_w, cy - half_h],  # 우상
            [cx + half_w, cy + half_h],  # 우하
            [cx - half_w, cy + half_h]   # 좌하
        ], dtype=np.float32)

    # 5. 사용자 클릭 (SRC 좌표)
    selector = PointSelector(frame, f"Select 4 Points for {args.camera}")
    src_points = selector.select_points()
    print(f"✅ 선택된 좌표:\n{src_points}")

    # 6. 맵 생성 (Combined Map)
    print("⏳ 맵 생성 중... (Combined Map)")
    map1, map2 = get_combined_map(K, D, src_points, dst_points, BEV_SIZE, RESOLUTION)

    # 7. 파일 저장 (.npy)
    # 나중에 surround_stitching_node에서 이 이름 그대로 불러옵니다.
    np.save(f"{args.camera}_map_x.npy", map1)
    np.save(f"{args.camera}_map_y.npy", map2)
    
    print(f"\n🎉 성공! '{args.camera}_map_x.npy'와 '{args.camera}_map_y.npy'가 저장되었습니다.")

    # 8. 결과 확인용 이미지 저장
    result_img = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(f"{args.camera}_result.jpg", result_img)
    print(f"   결과 이미지: {args.camera}_result.jpg 확인해보세요.")

    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()