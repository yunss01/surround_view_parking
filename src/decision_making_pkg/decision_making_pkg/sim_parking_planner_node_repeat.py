import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSDurabilityPolicy, QoSReliabilityPolicy
from enum import Enum
import numpy as np
import math
import time
import threading
import sys 
from std_msgs.msg import String
from sensor_msgs.msg import Image 
from cv_bridge import CvBridge    
import cv2                        
from interfaces_pkg.msg import BevPerceptionResult, PathPlanningResult
import csv
import os
from datetime import datetime
from nav_msgs.msg import Odometry

#---------------Variable Setting---------------
SUB_BEV_PERCEPTION_TOPIC = "bev_perception_result"
SUB_BEV_IMAGE_TOPIC = "/surround_view/image_raw" 
PUB_PARKING_PATH_TOPIC = "parking_path"
PUB_PARKING_STATUS_TOPIC = "parking_status"

# [설정] 원하는 주차 공간 번호 (1~4)
target_env = os.getenv('TARGET_SPOT_ID')
if target_env is not None:
    TARGET_SPOT_ID = int(target_env)
else:
    TARGET_SPOT_ID = 3
#----------------------------------------------

class ParkingState(Enum):
    READY = 0
    APPROACHING = 1
    SEARCHING = 2
    CALCULATING = 3
    ARC_1_SWING = 4
    ARC_2_REVERSE = 5
    ALIGNMENT = 6
    FINISHED = 7
    PARKING_FAILED = 99     # [NEW]
    
    # [Unparking States]
    UNPARKING_EXIT_SPOT = 8 
    UNPARKING_TURN = 9      
    UNPARKING_DEPARTURE = 10 

class ParkingPlannerNodeRepeat(Node):
    def __init__(self):
        super().__init__('parking_planner_node_repeat')
        self.bridge = CvBridge()

        self.BEV_CENTER_X = 850 / 2
        self.BEV_CENTER_Y = 1150 / 2
        self.max_steering = 7.0

        # 최신 튜닝 값 적용
        self.approach_duration_map = {1: 9.3, 2: 13.4, 3: 17.4, 4: 21.5}
        self.approach_speed = 50.0
        self.unpark_duration_map = {1: 18.0, 2: 14.5, 3: 10.0, 4: 6.0}
        self.arc2_finish_threshold_map = {1: 2.5, 2: 2.1, 3: 1.8, 4: 1.8}

        self.state = ParkingState.READY
        self.target_spot_id = TARGET_SPOT_ID 
        self.latest_bev_msg = None 
        self.latest_bev_image = None 
        self.state_start_time = None
        
        self.locked_target_angle = 0.0
        self.locked_target_pos = (0, 0)
        
        self.prev_steer = 0.0
        self.alignment_count = 0
        self.switch_angle_threshold = 30.0
        self.has_corrected_initial_angle = False
        
        self.unparking_timer_start = None
        
        # QoS & Subscribers
        self.qos = QoSProfile(reliability=QoSReliabilityPolicy.RELIABLE, history=QoSHistoryPolicy.KEEP_LAST, depth=1)
        
        self.img_sub = self.create_subscription(Image, SUB_BEV_IMAGE_TOPIC, self.img_callback, self.qos)
        self.bev_sub = self.create_subscription(BevPerceptionResult, SUB_BEV_PERCEPTION_TOPIC, self.bev_callback, self.qos)
        self.path_pub = self.create_publisher(PathPlanningResult, PUB_PARKING_PATH_TOPIC, self.qos)
        self.status_pub = self.create_publisher(String, PUB_PARKING_STATUS_TOPIC, self.qos)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, self.qos)
        
        self.current_odom = {'x':0.0, 'y':0.0, 'yaw':0.0} 
        self.IDEAL_POSES = {
            1: {'x': -3.263, 'y': 8.659, 'yaw': 90.0},
            2: {'x': -3.263,  'y': 5.302, 'yaw': 90.0},
            3: {'x': -3.263,  'y': 1.883, 'yaw': 90.0},
            4: {'x': -3.263,  'y': -1.452, 'yaw': 90.0}
        }

        self.start_arc1_angle = None
        
        # [Auto Start] 2초 뒤 자동 시작
        self.create_timer(6.0, self.auto_start)
        self.timer = self.create_timer(0.05, self.control_loop)
        
        self.get_logger().info(f"✅ Repeat Node Started. Will park at Spot {self.target_spot_id} automatically.")

    def auto_start(self):
        # 이미 시작했으면 패스
        if self.state != ParkingState.READY: return
        self.get_logger().info(f"🚀 Auto-Starting Sequence for Spot {self.target_spot_id}")
        self.set_target_spot(self.target_spot_id)

    def bev_callback(self, msg: BevPerceptionResult):
        self.latest_bev_msg = msg

    def img_callback(self, msg: Image):
        try:
            self.latest_bev_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            pass

    def set_target_spot(self, idx):
        self.target_spot_id = idx
        self.state_start_time = self.get_clock().now()
        self.smooth_heading_error = 0.0
        self.alignment_count = 0
        self.has_corrected_initial_angle = False
        self.evaluation_saved = False
        # [★추가할 초기화 변수들★] 이전에 쓰던 값이 남아있으면 안 됨!
        self.prev_steer = 0.0        # 조향각 초기화 (매우 중요)
        self.angle_history = []      # 각도 필터 초기화
        self.locked_target_pos = (0, 0) # 목표 좌표 초기화
        self.start_arc1_angle = None # 아크 시작 각도 초기화

        self.state = ParkingState.APPROACHING
        self.get_logger().info(f"State -> APPROACHING (Spot {idx})")

    def force_stop(self):
        self.state = ParkingState.READY
        self.publish_control(0.0, 0.0)

    def calculate_entrance_angle(self, obj):
        dx = obj.p2_bev.x - obj.p1_bev.x
        dy = obj.p2_bev.y - obj.p1_bev.y
        return math.degrees(math.atan2(dy, dx))

    # [최신 로직] 사이드 픽셀 검사 (ROI 450~600, Thresh 25.0)
    def check_side_occupancy(self):
        if self.latest_bev_image is None: 
            return True 
        
        h, w, _ = self.latest_bev_image.shape
        
        # 앞쪽으로 당겨진 ROI 좌표
        roi_x1, roi_x2 = 700, 850
        roi_y1, roi_y2 = 450, 600
        
        roi_x1 = max(0, min(roi_x1, w))
        roi_x2 = max(0, min(roi_x2, w))
        roi_y1 = max(0, min(roi_y1, h))
        roi_y2 = max(0, min(roi_y2, h))
        
        side_roi = self.latest_bev_image[roi_y1:roi_y2, roi_x1:roi_x2]
        if side_roi.size == 0: return False

        gray = cv2.cvtColor(side_roi, cv2.COLOR_BGR2GRAY)
        mean, stddev = cv2.meanStdDev(gray)
        std_val = stddev[0][0]
        
        self.get_logger().info(f"🧐 Side StdDev: {std_val:.2f}")
        return std_val > 35.0

    def control_loop(self):

        if self.state == ParkingState.READY: pass
        elif self.state == ParkingState.APPROACHING: self.run_approaching()
        elif self.state == ParkingState.SEARCHING: self.run_searching()
        elif self.state == ParkingState.CALCULATING: self.run_calculating()
        elif self.state == ParkingState.ARC_1_SWING: self.run_arc_1_swing()
        elif self.state == ParkingState.ARC_2_REVERSE: self.run_arc_2_reverse()
        elif self.state == ParkingState.ALIGNMENT: self.run_alignment()
        elif self.state == ParkingState.FINISHED: self.run_finished()
        elif self.state == ParkingState.PARKING_FAILED: self.run_parking_failed()
        
        elif self.state == ParkingState.UNPARKING_EXIT_SPOT: self.run_unparking_exit_spot()
        elif self.state == ParkingState.UNPARKING_TURN: self.run_unparking_turn()
        elif self.state == ParkingState.UNPARKING_DEPARTURE: self.run_unparking_departure()
        
        self.status_pub.publish(String(data=self.state.name))

    def run_approaching(self):
        elapsed = (self.get_clock().now() - self.state_start_time).nanoseconds / 1e9
        duration = self.approach_duration_map.get(self.target_spot_id, 5.0)
        if elapsed < duration: self.publish_control(0.0, self.approach_speed)
        else:
            self.publish_control(0.0, 0.0)
            time.sleep(0.5)
            self.state = ParkingState.SEARCHING

    def run_searching(self):
        # [1단계] 물리적 점유 확인 (Side Occupancy Check)
        if self.check_side_occupancy():
            self.get_logger().warn(f"🚫 Side Occupied! (StdDev High). Auto-skipping...")
            self.state = ParkingState.PARKING_FAILED
            return

        # [2단계] 비어 있다면 주차선 찾기
        target_obj = None
        if self.latest_bev_msg and self.latest_bev_msg.objects:
            candidates = [o for o in self.latest_bev_msg.objects 
                          if abs(o.center_bev.y - self.BEV_CENTER_Y) < 300]
            if candidates:
                target_obj = max(candidates, key=lambda o: o.center_bev.y)

        if target_obj:
            self.locked_target_angle = self.calculate_entrance_angle(target_obj)
            self.locked_target_pos = (target_obj.center_bev.x, target_obj.center_bev.y)
            
            self.state = ParkingState.CALCULATING
        else:
            self.get_logger().warn(f"❓ Clear Side, but NO Bbox detected. Abort.")
            self.state = ParkingState.PARKING_FAILED

    def run_calculating(self):
        target_obj = self.get_tracked_target()
        self.start_arc1_angle = self.calculate_entrance_angle(target_obj) if target_obj else -90.0
        self.state = ParkingState.ARC_1_SWING

    def run_arc_1_swing(self):
        target_obj = self.get_tracked_target()
        if target_obj is None:
            self.publish_control(-self.max_steering, 50.0)
            return
        current_angle = self.calculate_entrance_angle(target_obj)
        delta_angle = abs(current_angle - self.start_arc1_angle)
        if delta_angle > 180.0: delta_angle = 360.0 - delta_angle
        if delta_angle > self.switch_angle_threshold:
            self.publish_control(0.0, 0.0)
            time.sleep(0.5)
            self.state = ParkingState.ARC_2_REVERSE
            self.alignment_count = 0
            return
        self.publish_control(-self.max_steering, 50.0)

    def run_arc_2_reverse(self):
        target_obj = self.get_tracked_target()
        if target_obj is None:
            self.publish_control(0.0, -30.0, alpha=1.0) 
            return
        current_angle = self.calculate_entrance_angle(target_obj)
        target_threshold = self.arc2_finish_threshold_map.get(self.target_spot_id, 3.0)
        if abs(current_angle) < target_threshold: self.alignment_count += 1
        else: self.alignment_count = 0
        if self.alignment_count >= 3:
            self.prev_steer = 0.0 
            self.publish_control(0.0, 0.0, alpha=1.0) 
            time.sleep(1.5) 
            self.get_logger().info(f"Arc 2 Finished.")
            self.state = ParkingState.ALIGNMENT
            self.has_corrected_initial_angle = False 
            return
        self.publish_control(-self.max_steering, -30.0, alpha=0.3)

    def run_alignment(self):
        target_obj = self.get_tracked_target()
        if target_obj is None:
             self.publish_control(0.0, -30.0, alpha=1.0)
             return
        dist_y = abs(target_obj.center_bev.y - self.BEV_CENTER_Y)
        if dist_y < 70: 
            self.state = ParkingState.FINISHED
            self.publish_control(0.0, 0.0)
            self.get_logger().info(f"Success! Parking Finished (DistY: {dist_y:.1f}).")
            return
        if not self.has_corrected_initial_angle:
            current_angle = self.calculate_entrance_angle(target_obj)
            if abs(current_angle) > 1.0:
                correction = -current_angle * 0.5 
                correction = max(min(correction, 3.0), -3.0) 
                self.publish_control(correction, -30.0, alpha=1.0)
                return 
            else:
                self.has_corrected_initial_angle = True
        self.publish_control(0.0, -30.0, alpha=1.0)

    def run_finished(self):
        # 주차 완료 -> 데이터 저장 -> 2초 대기 -> 출차 시작
        self.publish_control(0.0, 0.0)
        time.sleep(1.0)
        if not getattr(self, 'evaluation_saved', False):
            self.save_evaluation_data()
            self.evaluation_saved = True
            self.get_logger().info("✅ Parking Done. Waiting 2s before Unparking...")
            
            sys.exit(0) # 프로그램 완전 종료

            # 2초 후 자동 출차 트리거
            #self.create_timer(2.0, self.trigger_auto_unpark)

    def trigger_auto_unpark(self):
        if self.state == ParkingState.FINISHED:
            self.get_logger().info("🔄 Auto Unparking Started...")
            self.state = ParkingState.UNPARKING_EXIT_SPOT

    def run_parking_failed(self):
        # 주차 실패(장애물) -> 1초 대기 -> 회피 주행
        self.publish_control(0.0, 0.0)
        self.get_logger().warn("⚠️ Parking Failed. Resuming path in 1s...")
        time.sleep(1.0)
        
        # 회피 주행 시작
        self.state = ParkingState.UNPARKING_DEPARTURE
        self.unparking_timer_start = self.get_clock().now()
        if self.target_spot_id is None: self.target_spot_id = 4

    def run_unparking_exit_spot(self):
        self.publish_control(0.0, 30.0)
        target_obj = self.get_tracked_target() 
        threshold_y = 900.0 
        if target_obj is None or target_obj.center_bev.y > threshold_y:
            self.get_logger().info("Target Spot Exited. Starting Turn...")
            self.state = ParkingState.UNPARKING_TURN

    def run_unparking_turn(self):
        self.publish_control(self.max_steering, 30.0)
        target_obj = self.get_tracked_target()
        if target_obj is not None:
            current_angle = self.calculate_entrance_angle(target_obj)
            if abs(current_angle) >= 88.0:
                self.get_logger().info(f"Turn Complete. Departing...")
                self.state = ParkingState.UNPARKING_DEPARTURE
                self.unparking_timer_start = self.get_clock().now()

    def run_unparking_departure(self):
        self.publish_control(0.0, 50.0)
        duration = self.unpark_duration_map.get(self.target_spot_id, 5.0)
        elapsed = (self.get_clock().now() - self.unparking_timer_start).nanoseconds / 1e9
        
        if elapsed > duration:
            self.get_logger().info("🏁 Simulation Sequence Finished. Exiting Node.")
            self.publish_control(0.0, 0.0)
            time.sleep(1.0)
            sys.exit(0) # 프로그램 완전 종료

    # Helper methods (get_tracked_target, publish_control, odom_callback, save_evaluation_data)
    # ... (기존과 동일하므로 아래에 포함) ...
    def get_tracked_target(self):
        if self.latest_bev_msg is None or not self.latest_bev_msg.objects: return None
        candidates = []
        ref_x, ref_y = self.locked_target_pos if self.locked_target_pos != (0,0) else (self.BEV_CENTER_X, self.BEV_CENTER_Y + 200)
        for obj in self.latest_bev_msg.objects:
            dist = math.hypot(obj.center_bev.x - ref_x, obj.center_bev.y - ref_y)
            candidates.append((dist, obj))
        candidates.sort(key=lambda x: x[0])
        if not candidates: return None
        if candidates[0][0] > 300.0: return None
        best_target = candidates[0][1]
        self.locked_target_pos = (best_target.center_bev.x, best_target.center_bev.y)
        return best_target

    def publish_control(self, steer, speed, alpha=0.7):
        filtered_steer = (alpha * steer) + ((1 - alpha) * self.prev_steer)
        self.prev_steer = filtered_steer
        msg = PathPlanningResult()
        msg.x_points = [float(filtered_steer)]
        msg.y_points = [float(speed)]
        self.path_pub.publish(msg)

    def odom_callback(self, msg: Odometry):
        self.current_odom['x'] = msg.pose.pose.position.x
        self.current_odom['y'] = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        yaw_rad = math.atan2(siny_cosp, cosy_cosp)
        self.current_odom['yaw'] = math.degrees(yaw_rad)

    def save_evaluation_data(self):
        if self.target_spot_id is None: return
        end_time = self.get_clock().now()
        duration = (end_time - self.state_start_time).nanoseconds / 1e9
        ideal = self.IDEAL_POSES.get(self.target_spot_id, {'x':0, 'y':0, 'yaw':0})
        dist_error_m = math.sqrt((self.current_odom['x'] - ideal['x'])**2 + (self.current_odom['y'] - ideal['y'])**2)
        total_dist_error_cm = dist_error_m * 100.0
        yaw_diff = abs(self.current_odom['yaw'] - ideal['yaw'])
        if yaw_diff > 180: yaw_diff = 360 - yaw_diff
        
        filename = f"parking_evaluation_spot_{self.target_spot_id}_obstacle2&4.csv"
        file_exists = os.path.isfile(filename)
        with open(filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(['Date', 'Target_Spot', 'Duration(s)', 'Total_Dist_Err(cm)', 'Heading_Err(deg)', 'Final_X', 'Final_Y', 'Final_Yaw'])
            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                self.target_spot_id,
                f"{duration:.2f}",
                f"{total_dist_error_cm:.2f}",
                f"{yaw_diff:.2f}",
                f"{self.current_odom['x']:.3f}",
                f"{self.current_odom['y']:.3f}",
                f"{self.current_odom['yaw']:.3f}"
            ])
        self.get_logger().info(f"📊 Evaluation Saved!")

def main(args=None):
    rclpy.init(args=args)
    node = ParkingPlannerNodeRepeat()
    try: rclpy.spin(node)
    except SystemExit: pass
    except KeyboardInterrupt: pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
