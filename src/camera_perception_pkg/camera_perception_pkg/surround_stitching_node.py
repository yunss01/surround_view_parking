import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import os
import message_filters
import yaml # 파라미터 저장용

# ==========================================
# 🎨 전체 캔버스 설정 (수정됨)
# ==========================================
CANVAS_WIDTH = 850
CANVAS_HEIGHT = 1150
CANVAS_CENTER_X = CANVAS_WIDTH // 2
CANVAS_CENTER_Y = CANVAS_HEIGHT // 2

# 🚗 차량(검은 박스) 크기
CAR_WIDTH = 240
CAR_HEIGHT = 460

# 저장할 파라미터 파일 이름
PARAM_FILE = "surround_view_param.yaml"

class StitchingNode(Node):
    def __init__(self):
        super().__init__('stitching_node')
        self.bridge = CvBridge()
        
        # 1. 맵 파일 로드
        self.maps = {}
        if not self.load_maps():
            self.get_logger().error("❌ .npy 맵 파일 로드 실패!")

        # 2. 토픽 구독
        self.sub_f = message_filters.Subscriber(self, Image, '/fo_camera/image_raw')
        self.sub_b = message_filters.Subscriber(self, Image, '/rear_camera/image_raw')
        self.sub_l = message_filters.Subscriber(self, Image, '/left_camera/image_raw')
        self.sub_r = message_filters.Subscriber(self, Image, '/right_camera/image_raw')

        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.sub_f, self.sub_b, self.sub_l, self.sub_r], queue_size=10, slop=0.2
        )
        self.ts.registerCallback(self.callback)
        
        self.pub = self.create_publisher(Image, '/surround_view/image_raw', 10)
        
        # ==========================================
        # 🎛️ GUI 초기화
        # ==========================================
        self.cam_names = ['front', 'back', 'left', 'right']
        self.curr_cam_idx = 0 
        
        # 기본값 설정
        self.cam_params = {
            'front': {'x': 0, 'y': -350, 'r': 0, 'ct': 0, 'cb': 0, 'cl': 0, 'cr': 0},
            'back':  {'x': 0, 'y':  350, 'r': 0, 'ct': 0, 'cb': 0, 'cl': 0, 'cr': 0},
            'left':  {'x': -250, 'y': 0, 'r': 0, 'ct': 0, 'cb': 0, 'cl': 0, 'cr': 0},
            'right': {'x':  250, 'y': 0, 'r': 0, 'ct': 0, 'cb': 0, 'cl': 0, 'cr': 0}
        }

        # 저장된 파라미터가 있으면 불러오기
        self.load_params()

        cv2.namedWindow("Tuning Panel", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Tuning Panel", 400, 600)
        
        # 트랙바 생성
        cv2.createTrackbar("Select Cam", "Tuning Panel", 0, 3, self.nothing)
        cv2.createTrackbar("Pos X", "Tuning Panel", 500, 1000, self.nothing)
        cv2.createTrackbar("Pos Y", "Tuning Panel", 500, 1000, self.nothing)
        cv2.createTrackbar("Rotation", "Tuning Panel", 100, 200, self.nothing)
        cv2.createTrackbar("Crop Top",    "Tuning Panel", 0, 200, self.nothing)
        cv2.createTrackbar("Crop Bottom", "Tuning Panel", 0, 200, self.nothing)
        cv2.createTrackbar("Crop Left",   "Tuning Panel", 0, 200, self.nothing)
        cv2.createTrackbar("Crop Right",  "Tuning Panel", 0, 200, self.nothing)

        # 시작 시 현재 카메라(Front) 값으로 슬라이더 동기화
        self.update_trackbars('front')

        self.get_logger().info("✅ Stitching Node Started! Press 's' to Save.")

    def nothing(self, x): pass

    def load_maps(self):
        map_dir = '/home/autolab/surround_view_parking-main/npy'
        try:
            self.maps['front_x'] = np.load(os.path.join(map_dir, 'front_map_x.npy'))
            self.maps['front_y'] = np.load(os.path.join(map_dir, 'front_map_y.npy'))
            self.maps['back_x']  = np.load(os.path.join(map_dir, 'rear_map_x.npy'))
            self.maps['back_y']  = np.load(os.path.join(map_dir, 'rear_map_y.npy'))
            self.maps['left_x']  = np.load(os.path.join(map_dir, 'left_map_x.npy'))
            self.maps['left_y']  = np.load(os.path.join(map_dir, 'left_map_y.npy'))
            self.maps['right_x'] = np.load(os.path.join(map_dir, 'right_map_x.npy'))
            self.maps['right_y'] = np.load(os.path.join(map_dir, 'right_map_y.npy'))
            return True
        except Exception as e:
            self.get_logger().error(f"❌ 맵 로드 실패: {e}")
            return False

    def load_params(self):
        """ yaml 파일에서 파라미터 불러오기 """
        if os.path.exists(PARAM_FILE):
            try:
                with open(PARAM_FILE, 'r') as f:
                    data = yaml.safe_load(f)
                    if data:
                        self.cam_params = data
                        self.get_logger().info(f"📂 파라미터 로드 완료: {PARAM_FILE}")
            except Exception as e:
                self.get_logger().warn(f"⚠️ 파라미터 로드 실패: {e}")

    def save_params(self):
        """ 현재 파라미터를 yaml 파일로 저장 """
        try:
            with open(PARAM_FILE, 'w') as f:
                yaml.dump(self.cam_params, f)
            self.get_logger().info(f"💾 파라미터 저장 완료! -> {PARAM_FILE}")
        except Exception as e:
            self.get_logger().error(f"❌ 저장 실패: {e}")

    def update_trackbars(self, cam_name):
        """ 선택된 카메라의 값으로 슬라이더 위치 갱신 """
        p = self.cam_params[cam_name]
        cv2.setTrackbarPos("Pos X", "Tuning Panel", p['x'] + 500)
        cv2.setTrackbarPos("Pos Y", "Tuning Panel", p['y'] + 500)
        cv2.setTrackbarPos("Rotation", "Tuning Panel", int(p['r']*2) + 100)
        cv2.setTrackbarPos("Crop Top", "Tuning Panel", p['ct'])
        cv2.setTrackbarPos("Crop Bottom", "Tuning Panel", p['cb'])
        cv2.setTrackbarPos("Crop Left", "Tuning Panel", p['cl'])
        cv2.setTrackbarPos("Crop Right", "Tuning Panel", p['cr'])

    def process_image(self, img, name, params):
        if f'{name}_x' not in self.maps: return img

        # 1. Remap
        bev = cv2.remap(img, self.maps[f'{name}_x'], self.maps[f'{name}_y'], 
                        interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        
        # 2. Crop
        h, w = bev.shape[:2]
        ct, cb = params['ct'], params['cb']
        cl, cr = params['cl'], params['cr']
        
        if ct + cb < h and cl + cr < w:
            bev = bev[ct : h-cb, cl : w-cr]
        
        # 3. Rotate 90
        if name == 'back':
            bev = cv2.rotate(bev, cv2.ROTATE_180)
        elif name == 'left':
            bev = cv2.rotate(bev, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif name == 'right':
            bev = cv2.rotate(bev, cv2.ROTATE_90_CLOCKWISE)
            
        # 4. Fine Rotate
        if params['r'] != 0:
            h, w = bev.shape[:2]
            M = cv2.getRotationMatrix2D((w//2, h//2), params['r'], 1.0)
            bev = cv2.warpAffine(bev, M, (w, h), borderMode=cv2.BORDER_CONSTANT)
            
        return bev

    def place_image(self, canvas, img, x_offset, y_offset):
        h, w = img.shape[:2]
        start_x = CANVAS_CENTER_X + x_offset - (w // 2)
        start_y = CANVAS_CENTER_Y + y_offset - (h // 2)
        end_x = start_x + w
        end_y = start_y + h

        visible_start_x = max(0, start_x)
        visible_start_y = max(0, start_y)
        visible_end_x = min(CANVAS_WIDTH, end_x)
        visible_end_y = min(CANVAS_HEIGHT, end_y)
        
        if visible_end_x <= visible_start_x or visible_end_y <= visible_start_y:
            return canvas
        
        crop_start_x = visible_start_x - start_x
        crop_start_y = visible_start_y - start_y
        crop_end_x = crop_start_x + (visible_end_x - visible_start_x)
        crop_end_y = crop_start_y + (visible_end_y - visible_start_y)

        roi = img[crop_start_y:crop_end_y, crop_start_x:crop_end_x]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        mask = gray > 10 
        
        canvas_area = canvas[visible_start_y:visible_end_y, visible_start_x:visible_end_x]
        canvas_area[mask] = roi[mask]
        canvas[visible_start_y:visible_end_y, visible_start_x:visible_end_x] = canvas_area
        
        return canvas

    def callback(self, f, b, l, r):
        try:
            # ----------------------------------------------------
            # 🎛️ GUI Logic
            # ----------------------------------------------------
            selected_idx = cv2.getTrackbarPos("Select Cam", "Tuning Panel")
            
            # 카메라 변경 감지
            if selected_idx != self.curr_cam_idx:
                self.curr_cam_idx = selected_idx
                target_name = self.cam_names[selected_idx]
                self.update_trackbars(target_name)
            
            # 현재 값 읽어서 업데이트
            else:
                target_name = self.cam_names[self.curr_cam_idx]
                
                x_val = cv2.getTrackbarPos("Pos X", "Tuning Panel")
                y_val = cv2.getTrackbarPos("Pos Y", "Tuning Panel")
                r_val = cv2.getTrackbarPos("Rotation", "Tuning Panel")
                
                self.cam_params[target_name]['x'] = x_val - 500
                self.cam_params[target_name]['y'] = y_val - 500
                self.cam_params[target_name]['r'] = (r_val - 100) * 0.5
                
                self.cam_params[target_name]['ct'] = cv2.getTrackbarPos("Crop Top", "Tuning Panel")
                self.cam_params[target_name]['cb'] = cv2.getTrackbarPos("Crop Bottom", "Tuning Panel")
                self.cam_params[target_name]['cl'] = cv2.getTrackbarPos("Crop Left", "Tuning Panel")
                self.cam_params[target_name]['cr'] = cv2.getTrackbarPos("Crop Right", "Tuning Panel")

            # ----------------------------------------------------
            # Process & Stitch
            # ----------------------------------------------------
            imgs = {
                'front': self.bridge.imgmsg_to_cv2(f, "bgr8"),
                'back':  self.bridge.imgmsg_to_cv2(b, "bgr8"),
                'left':  self.bridge.imgmsg_to_cv2(l, "bgr8"),
                'right': self.bridge.imgmsg_to_cv2(r, "bgr8")
            }
            
            bevs = {}
            for name in self.cam_names:
                bevs[name] = self.process_image(imgs[name], name, self.cam_params[name])

            final_canvas = np.zeros((CANVAS_HEIGHT, CANVAS_WIDTH, 3), dtype=np.uint8)
            
            # 합성 순서: Left/Right -> Front/Back
            final_canvas = self.place_image(final_canvas, bevs['left'],  self.cam_params['left']['x'],  self.cam_params['left']['y'])
            final_canvas = self.place_image(final_canvas, bevs['right'], self.cam_params['right']['x'], self.cam_params['right']['y'])
            final_canvas = self.place_image(final_canvas, bevs['front'], self.cam_params['front']['x'], self.cam_params['front']['y'])
            final_canvas = self.place_image(final_canvas, bevs['back'],  self.cam_params['back']['x'],  self.cam_params['back']['y'])

            # 차량 박스
            cv2.rectangle(final_canvas, 
                          (CANVAS_CENTER_X - CAR_WIDTH // 2, CANVAS_CENTER_Y - CAR_HEIGHT // 2), 
                          (CANVAS_CENTER_X + CAR_WIDTH // 2, CANVAS_CENTER_Y + CAR_HEIGHT // 2), 
                          (0, 0, 0), -1)

            self.pub.publish(self.bridge.cv2_to_imgmsg(final_canvas, "bgr8"))
            
            # 화면 표시
            view = cv2.resize(final_canvas, (600, int(600 * CANVAS_HEIGHT / CANVAS_WIDTH))) 
            cv2.imshow("Surround View (Tuning)", view)
            
            # [키 입력 저장] 's' 누르면 저장
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                self.save_params()
            
        except Exception as e:
            self.get_logger().error(f"처리 에러: {e}")

def main():
    rclpy.init()
    node = StitchingNode()
    try: rclpy.spin(node)
    except: pass
    cv2.destroyAllWindows()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()