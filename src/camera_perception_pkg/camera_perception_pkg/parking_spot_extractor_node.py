import cv2
import rclpy
import numpy as np
import math
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSDurabilityPolicy, QoSReliabilityPolicy
from cv_bridge import CvBridge
from std_msgs.msg import String

# 사용할 메시지 타입
from sensor_msgs.msg import Image
from interfaces_pkg.msg import DetectionArray, BevObject, BevPerceptionResult, Point2D

#---------------Variable Setting---------------
# Subscribe할 토픽 이름
SUB_TOPIC_NAME = "detections"

# Publish할 토픽 이름
PUB_TOPIC_NAME = "bev_perception_result"
DEBUG_IMAGE_TOPIC_NAME = "bev_perception_image"

# 화면에 이미지를 처리하는 과정을 띄울것인지 여부: True, 또는 False 중 택1
SHOW_IMAGE = True

# 물리적 수치 설정
CAR_LENGTH_M = 2.86      # 차량 전체 길이 (m)
CAR_LENGTH_PX = 340.0    # 서라운드뷰 상 차량 길이 (pixels)
PARKING_DEPTH_M = 1.58  # 입구에서 중심점까지의 거리 (3.16m / 2)
#----------------------------------------------

# 1픽셀당 미터 (Scale Factor)
METER_PER_PIXEL = CAR_LENGTH_M / CAR_LENGTH_PX 

class BevPerceptionExtractor(Node):
    def __init__(self):
        super().__init__('bev_perception_extractor_node')

        # 파라미터 설정
        self.sub_topic = self.declare_parameter('sub_detection_topic', SUB_TOPIC_NAME).value
        self.pub_topic = self.declare_parameter('pub_topic', PUB_TOPIC_NAME).value
        self.show_image = self.declare_parameter('show_image', SHOW_IMAGE).value

        self.cv_bridge = CvBridge()
        self.qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE, 
            history=QoSHistoryPolicy.KEEP_LAST, 
            durability=QoSDurabilityPolicy.VOLATILE, 
            depth=1
        )
        
        # YOLO 결과 구독
        self.subscriber = self.create_subscription(
            DetectionArray, self.sub_topic, self.detections_callback, self.qos_profile)

        # 이전 프레임의 키포인트 저장을 위한 변수
        self.prev_keypoints = {} # ID별 혹은 단일 객체용
        self.tracked_spots = [] # 이전 프레임의 주차칸 벡터 정보를 저장할 메모리

        # 결과 발행
        self.publisher = self.create_publisher(BevPerceptionResult, self.pub_topic, self.qos_profile)
        self.debug_image_publisher = self.create_publisher(Image, DEBUG_IMAGE_TOPIC_NAME, self.qos_profile)
        
        self.get_logger().info("✅ BEV Perception Extractor Node (Individual Processing) has started.")

    def detections_callback(self, detection_msg: DetectionArray):
        bev_objects = []
        
        if not detection_msg.detections:
            empty_msg = BevPerceptionResult()
            empty_msg.header = detection_msg.header
            self.publisher.publish(empty_msg)
            return

        # 1. 캔버스 크기 설정
        img_h = detection_msg.detections[0].mask.height if detection_msg.detections[0].mask.height > 0 else 1024
        img_w = detection_msg.detections[0].mask.width if detection_msg.detections[0].mask.width > 0 else 872
        
        # 디버깅용 전체 이미지 (시각화용)
        debug_canvas = np.zeros((img_h, img_w, 3), dtype=np.uint8)

        # 이미지 중심 (내 차 위치)
        img_center_x = img_w / 2.0
        img_center_y = img_h / 2.0

        current_frame_spots = [] # 이번 프레임에서 확정된 스팟 정보 저장용

        # --- 2. 개별 객체 처리 (Loop) ---
        # 모든 마스크를 한 곳에 그리지 않고, 하나씩 따로 분석합니다.
        for detection in detection_msg.detections:
            if detection.class_name != 'parking_spot':
                continue

            kpts = detection.keypoints.data
            
            # 키포인트가 2개가 아니면 스킵 (입구만 알면 됨)
            if len(kpts) < 2:
                continue

            # 좌표 추출 (Point2D 객체로 가정)
            p1 = [kpts[0].point.x, kpts[0].point.y] # 입구 왼쪽
            p2 = [kpts[1].point.x, kpts[1].point.y] # 입구 오른쪽

  
            # [복원 로직] 입구(p1, p2)는 살아있는데, 안쪽(p3, p4)이 죽었거나(0,0) 신뢰도가 낮으면 복구한다.
            # 조건: p1, p2는 정상(5.0 이상)이어야 함
            if p1[0] < 5.0 and p2[0] < 5.0:
                continue
 
            # --- 로직 1: 입구 중심점 계산 ---
            # 0번과 1번 점 사이가 무조건 입구입니다.
            entrance_cx = (p1[0] + p2[0]) / 2.0
            entrance_cy = (p1[1] + p2[1]) / 2.0

            # --- 로직 2: 진입 방향 벡터 계산 ---
            # 1. 일단 p1 -> p2 기준의 수직 벡터를 구합니다.
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            
            # 수직 벡터 후보 (후보 A: -dy, dx)
            vec_x = -dy
            vec_y = dx

            # 2. [핵심] 시간적 연속성(Temporal Consistency) 적용
            # 이 스팟이 이전 프레임의 어떤 스팟과 같은 녀석인지 찾습니다.
            match_found = False
            last_vec = (0, 0)

            for old_spot in self.tracked_spots:
                old_cx, old_cy = old_spot['center']
                # 거리가 가까우면(예: 100픽셀 이내) 같은 스팟으로 간주
                dist = math.hypot(entrance_cx - old_cx, entrance_cy - old_cy)
                if dist < 100.0:
                    last_vec = old_spot['vector']
                    match_found = True
                    break
            
            if match_found:
                # [Case A] 추적 중인 스팟: "이전 방향과 비슷하게 맞춰라"
                # 내적(Dot Product)을 통해 방향 확인
                dot_prod = (vec_x * last_vec[0]) + (vec_y * last_vec[1])
                
                # 내적이 음수면 방향이 180도 뒤집힌 것이므로 강제 반전
                if dot_prod < 0:
                    vec_x = -vec_x
                    vec_y = -vec_y
            else:
                # [Case B] 처음 본 스팟: "Distance Logic" 사용 (멀리 있을 땐 이게 정확함)
                test_x = entrance_cx + vec_x
                test_y = entrance_cy + vec_y
                dist_to_car = (entrance_cx - img_center_x)**2 + (entrance_cy - img_center_y)**2
                dist_moved = (test_x - img_center_x)**2 + (test_y - img_center_y)**2
                
                if dist_moved < dist_to_car: # 가까워지면 반대 방향이 정답
                    vec_x = -vec_x
                    vec_y = -vec_y

            # 3. 계산된 확정 벡터를 저장 (다음 프레임을 위해)
            current_frame_spots.append({
                'center': (entrance_cx, entrance_cy),
                'vector': (vec_x, vec_y)
            })

            # 벡터 정규화 (Unit Vector)
            mag = math.hypot(vec_x, vec_y)
            if mag == 0: mag = 1.0
            unit_x = vec_x / mag
            unit_y = vec_y / mag

            # 주차칸 전체 깊이 (약 3.16m에 해당하는 픽셀)
            # PARKING_DEPTH_M은 절반(중심까지)이므로 2배 해줌
            full_depth_px = (PARKING_DEPTH_M * 2.0) / METER_PER_PIXEL 
            
            # 가상의 안쪽 점(p3, p4) 생성
            p4 = [0.0, 0.0] # p4는 p1(왼쪽 입구) 뒤
            p3 = [0.0, 0.0] # p3는 p2(오른쪽 입구) 뒤
            
            p4[0] = p1[0] + (unit_x * full_depth_px)
            p4[1] = p1[1] + (unit_y * full_depth_px)
            
            p3[0] = p2[0] + (unit_x * full_depth_px)
            p3[1] = p2[1] + (unit_y * full_depth_px)

            # 최종 타겟 중심점 (입구에서 절반 깊이만큼 들어간 곳)
            half_depth_px = PARKING_DEPTH_M / METER_PER_PIXEL
            final_cx = entrance_cx + (unit_x * half_depth_px)
            final_cy = entrance_cy + (unit_y * half_depth_px)

            angle_rad = math.atan2(dy, dx)
            angle_deg = math.degrees(angle_rad) # 이게 Planner에서 쓸 각도

            # 결과 객체 생성
            bev_obj = BevObject()
            bev_obj.class_name = 'parking_spot'
            bev_obj.center_bev.x = float(final_cx)
            bev_obj.center_bev.y = float(final_cy)
            bev_obj.angle_bev = float(angle_deg) 
            
            # 꼭짓점 정보 저장 (Planner에서 p1, p2를 입구로 사용함)
            bev_obj.p1_bev.x = float(p1[0])
            bev_obj.p1_bev.y = float(p1[1])
            bev_obj.p2_bev.x = float(p2[0])
            bev_obj.p2_bev.y = float(p2[1])
            bev_obj.p3_bev.x = float(p3[0])
            bev_obj.p3_bev.y = float(p3[1])
            bev_obj.p4_bev.x = float(p4[0])
            bev_obj.p4_bev.y = float(p4[1])

            bev_objects.append(bev_obj)

            # --- 시각화 (Debugging) ---
            # 1. 뼈대 그리기 (Cyan 색상)
            pts = np.array([p1, p2, p3, p4], np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(debug_canvas, [pts], True, (255, 255, 0), 2) # Cyan Box
            
            # 2. 입구 라인 강조 (Blue)
            cv2.line(debug_canvas, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (255, 0, 0), 3)
            
            # 3. 입구 중심점 (Yellow)
            cv2.circle(debug_canvas, (int(entrance_cx), int(entrance_cy)), 5, (0, 255, 255), -1)
            
            # 4. 최종 타겟점 (Red)
            cv2.circle(debug_canvas, (int(final_cx), int(final_cy)), 7, (0, 0, 255), -1)
            
            # 5. 방향 벡터 (입구 -> 타겟)
            cv2.arrowedLine(debug_canvas, 
                            (int(entrance_cx), int(entrance_cy)), 
                            (int(final_cx), int(final_cy)), 
                            (0, 255, 0), 2) # Green Arrow
        
        # 이번 프레임의 결과를 전역 변수에 업데이트
        self.tracked_spots = current_frame_spots

        # --- 3. 결과 발행 ---
        result_msg = BevPerceptionResult()
        result_msg.header = detection_msg.header
        result_msg.objects = bev_objects
        self.publisher.publish(result_msg)

        # --- 4. 디버깅 이미지 발행 ---
        if self.show_image or self.debug_image_publisher.get_subscription_count() > 0:
            self.debug_image_publisher.publish(self.cv_bridge.cv2_to_imgmsg(debug_canvas, "bgr8"))
            
            if self.show_image:
                win_name = 'BEV Keypoint Result'
                cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(win_name, 600, 600)
                cv2.imshow(win_name, debug_canvas)
                cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = BevPerceptionExtractor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()
  
if __name__ == '__main__':
    main()