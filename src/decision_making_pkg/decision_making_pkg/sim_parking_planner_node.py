import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSDurabilityPolicy, QoSReliabilityPolicy
from enum import Enum
import numpy as np
import math
import time
import threading
import tkinter as tk
from std_msgs.msg import String
from interfaces_pkg.msg import BevPerceptionResult, PathPlanningResult
import csv  # 엑셀(CSV) 저장용
import os   # 파일 경로 확인용
from datetime import datetime
from nav_msgs.msg import Odometry # 오도메트리 메시지 타입
from geometry_msgs.msg import Quaternion # 쿼터니언 변환용

#---------------Variable Setting---------------
SUB_BEV_PERCEPTION_TOPIC = "bev_perception_result"
PUB_PARKING_PATH_TOPIC = "parking_path"
PUB_PARKING_STATUS_TOPIC = "parking_status"
#----------------------------------------------

class ParkingState(Enum):
    READY = 0               
    APPROACHING = 1         
    SEARCHING = 2           
    CALCULATING = 3         # [New] Two-Arc 경로 계산 및 타겟 고정 (Target Locking)
    ARC_1_SWING = 4         # [New] 제1 원호: 좌회전 전진
    ARC_2_REVERSE = 5       # [New] 제2 원호: 우회전 후진 (Turn-in)
    ALIGNMENT = 6           # [New] 잔여 오차 수정 (Visual Servoing)
    FINISHED = 7            

class ParkingGUI:
    def __init__(self, ros_node):
        self.node = ros_node
        self.root = tk.Tk()
        self.root.title("Two-Arc Parking")
        self.root.geometry("300x350")
        tk.Label(self.root, text="Two-Arc Path Planner", font=("Arial", 14, "bold")).pack(pady=10)
        
        self.spot_buttons = {}
        for i in range(1, 5):
            btn = tk.Button(self.root, text=f"Park at Spot {i}", 
                            font=("Arial", 12), height=2, width=20, bg="white",
                            command=lambda idx=i: self.select_spot(idx))
            btn.pack(pady=5)
            self.spot_buttons[i] = btn
            
        tk.Button(self.root, text="EMERGENCY STOP", font=("Arial", 12, "bold"), bg="red", fg="white",
                  command=self.node.force_stop).pack(pady=20)

    def close(self):
        self.root.destroy()

    def select_spot(self, idx):
        self.node.get_logger().info(f"GUI: Target Spot {idx} Selected!")
        self.node.set_target_spot(idx)
        for i, btn in self.spot_buttons.items():
            if i == idx: btn.config(bg="#4CAF50", fg="white")
            else: btn.config(bg="white", fg="black")
        self.root.update()

class ParkingPlannerNode2(Node):
    def __init__(self):
        super().__init__('parking_planner_node_2')

        self.BEV_CENTER_X = 850 / 2
        self.BEV_CENTER_Y = 1150 / 2
        self.max_steering = 7.0 # 하드웨어 최대 조향각

        # 접근 시간 (튜닝 필요)
        self.approach_duration_map = {1: 9.6, 2: 13.6, 3: 17.6, 4: 21.6}
        self.approach_speed = 50.0

        # 주차칸별 Arc 2 종료 각도 오차
        self.arc2_finish_threshold_map = {1: 3.0, 2: 3.0, 3: 2.8, 4: 2.0}

        self.state = ParkingState.READY
        self.target_spot_id = None
        self.latest_bev_msg = None 
        self.state_start_time = None
        
        # [Target Locking Variables]
        self.locked_target_angle = 0.0  # 고정된 타겟 각도
        self.locked_target_pos = (0, 0) # 고정된 타겟 위치 (상대좌표 아님, 초기 기준)
        
        # [Dual Smoothing Variables]
        self.smooth_heading_error = 0.0
        self.prev_steer = 0.0
        
        # [Two-Arc Params]
        self.switch_angle_threshold = 30.0 # 1차 회전 목표 각도 (Swing Out 각도)
        
        self.qos = QoSProfile(reliability=QoSReliabilityPolicy.RELIABLE, history=QoSHistoryPolicy.KEEP_LAST, depth=1)
        self.bev_sub = self.create_subscription(BevPerceptionResult, SUB_BEV_PERCEPTION_TOPIC, self.bev_callback, self.qos)
        self.path_pub = self.create_publisher(PathPlanningResult, PUB_PARKING_PATH_TOPIC, self.qos)
        self.status_pub = self.create_publisher(String, PUB_PARKING_STATUS_TOPIC, self.qos)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, self.qos)
        self.current_odom = {'x':0.0, 'y':0.0, 'yaw':0.0} # (x, y, yaw) 저장용
        self.IDEAL_POSES = {
            1: {'x': -3.15, 'y': 8.693, 'yaw': 90.0},
            2: {'x': -3.15,  'y': 5.302, 'yaw': 90.0},
            3: {'x': -3.15,  'y': 1.945, 'yaw': 90.0},
            4: {'x': -3.15,  'y': -1.411, 'yaw': 90.0}
        }

        # ARC 1 시작 시점의 각도를 저장할 변수
        self.start_arc1_angle = None
        # 한 번 직진 모드가 되면 절대 안 풀리게 하는 변수
        self.force_straight_active = False

        self.timer = self.create_timer(0.05, self.control_loop)
        self.get_logger().info("✅ Two-Arc Parking Planner (Node 2) Started.")

    def bev_callback(self, msg: BevPerceptionResult):
        self.latest_bev_msg = msg

    def set_target_spot(self, idx):
        self.target_spot_id = idx
        self.state_start_time = self.get_clock().now()
        self.smooth_heading_error = 0.0
        self.alignment_count = 0
        self.evaluation_saved = False
        self.state = ParkingState.APPROACHING
        self.get_logger().info(f"Start Approaching Spot {idx}...")

    def force_stop(self):
        self.state = ParkingState.READY
        self.publish_control(0.0, 0.0)
        self.get_logger().warn("EMERGENCY STOP TRIGGERED")

    def calculate_entrance_angle(self, obj):
        dx = obj.p2_bev.x - obj.p1_bev.x
        dy = obj.p2_bev.y - obj.p1_bev.y
        angle_rad = math.atan2(dy, dx)
        angle_deg = math.degrees(angle_rad)

        return angle_deg

    def control_loop(self):
        if self.state == ParkingState.READY: pass
        elif self.state == ParkingState.APPROACHING: self.run_approaching()
        elif self.state == ParkingState.SEARCHING: self.run_searching()
        elif self.state == ParkingState.CALCULATING: self.run_calculating()
        elif self.state == ParkingState.ARC_1_SWING: self.run_arc_1_swing()
        elif self.state == ParkingState.ARC_2_REVERSE: self.run_arc_2_reverse()
        elif self.state == ParkingState.ALIGNMENT: self.run_alignment()
        elif self.state == ParkingState.FINISHED: self.run_finished()
        
        self.status_pub.publish(String(data=self.state.name))

    # --- 1. 접근 (Open Loop) ---
    def run_approaching(self):
        elapsed = (self.get_clock().now() - self.state_start_time).nanoseconds / 1e9
        duration = self.approach_duration_map.get(self.target_spot_id, 5.0)
        if elapsed < duration: self.publish_control(0.0, self.approach_speed)
        else:
            self.publish_control(0.0, 0.0)
            time.sleep(0.5)
            self.state = ParkingState.SEARCHING

    # --- 2. 탐색 (Vision) ---
    def run_searching(self):
        if self.latest_bev_msg is None or not self.latest_bev_msg.objects:
            self.get_logger().warn("Searching...", throttle_duration_sec=1)
            return

        try:
            # 가장 가까운(화면 아래쪽) 주차칸 찾기
            target_obj = max(self.latest_bev_msg.objects, key=lambda o: o.center_bev.y)
            
            # [Target Locking] 초기 발견 시점의 정보를 저장
            self.locked_target_angle = self.calculate_entrance_angle(target_obj)
            self.locked_target_pos = (target_obj.center_bev.x, target_obj.center_bev.y)
            
            self.get_logger().info(f"Spot Found! Angle={self.locked_target_angle:.2f}. Locking Target.")
            self.state = ParkingState.CALCULATING
        except ValueError:
            return

    # --- 3. 경로 계산 (Two-Arc Logic) ---
    def run_calculating(self):
        """
        논문의 Two-Arc 원리를 적용:
        - 현재 위치에서 1차 회전(Swing Out)을 얼마나 해야 2차 회전(Reverse)으로 딱 들어갈지 계산.
        - 오도메트리가 없으므로, '각도'를 기준으로 스위칭 포인트를 잡음.
        """
        target_obj = self.get_tracked_target()
        if target_obj:
            # 현재 각도를 '시작 각도'로 저장
            self.start_arc1_angle = self.calculate_entrance_angle(target_obj)
            self.get_logger().info(f"Reference Angle Saved: {self.start_arc1_angle:.2f}")
        else:
            # 타겟이 없으면 90도로 가정 (Fallback)
            self.start_arc1_angle = -90.0
        
        self.get_logger().info(f"Path Calculated. Switch Angle: {self.switch_angle_threshold} deg.")
        self.state = ParkingState.ARC_1_SWING

    # --- 4. 제1 원호 (Swing Out) ---
    def run_arc_1_swing(self):
        """좌회전 전진하며 45도 각도 만들기"""
        target_obj = self.get_tracked_target()
        
        # 타겟을 놓쳐도 기존 궤적(좌회전) 유지 (Blind Mode)
        if target_obj is None:
            self.publish_control(-self.max_steering, 50.0)
            self.get_logger().info("Blind Swing...", throttle_duration_sec=1)
            return

        current_angle = self.calculate_entrance_angle(target_obj)
        delta_angle = abs(current_angle - self.start_arc1_angle)
        if delta_angle > 180.0:
            delta_angle = 360.0 - delta_angle
        
        # 시작 각도: 약 90도 (또는 -90도)
        # 목표 회전량(30도)을 넘었는지 확인
        if delta_angle > self.switch_angle_threshold:
            self.publish_control(0.0, 0.0)
            time.sleep(0.5)
            self.get_logger().info(f"Arc 1 Finished. Rotated {delta_angle:.1f} deg. Ready to Reverse.")
            self.state = ParkingState.ARC_2_REVERSE
            return

        self.publish_control(-self.max_steering, 50.0) # 좌회전 전진

    # --- 5. 제2 원호 (Reverse Entry) ---
    def run_arc_2_reverse(self):
        """우회전 후진으로 진입 (Turn-in)"""
        target_obj = self.get_tracked_target()

        if target_obj is None:
            self.publish_control(-self.max_steering, -30.0) # 궤적 유지
            return

        current_angle = self.calculate_entrance_angle(target_obj)

        # 현재 타겟 번호에 맞는 허용 오차 가져오기 (기본값 3.0)
        target_threshold = self.arc2_finish_threshold_map.get(self.target_spot_id, 3.0)
        
        # 한 번의 튐으로 넘어가지 않도록, 5번 연속 만족해야 넘어감
        if abs(current_angle) < target_threshold:
            self.alignment_count += 1
        else:
            self.alignment_count = 0 # 조건 깨지면 리셋

        if self.alignment_count >= 5: # 0.25초 이상 유지되면 진짜 정렬된 것
            self.publish_control(0.0, 0.0, alpha=1.0) # 즉시 정지
            time.sleep(0.5)
            self.get_logger().info(f"Arc 2 Finished. Aligned ({current_angle:.1f}). Fine Tuning...")
            self.state = ParkingState.ALIGNMENT
            return

        self.publish_control(-self.max_steering, -30.0) # 우회전 후진

    # --- 6. 정렬 (Hybrid Dual Smoothing) ---
    def run_alignment(self):
        target_obj = self.get_tracked_target()
        
        # 타겟 거리 확인용 (조향용 아님)
        if target_obj is None:
            # 타겟을 놓쳐도 핸들이 0이므로 그냥 뒤로 조금 더 가도 안전함
            # 하지만 안전을 위해 정지하거나, 아주 천천히 후진
            self.publish_control(0.0, 0.0)
            self.get_logger().warn("Target lost in Blind Reverse. Stopping.")
            return

        # 거리만 계산 (y축 기준)
        dist_y = abs(target_obj.center_bev.y - self.BEV_CENTER_Y)

        # [종료 조건] 거리가 충분히 가까워지면 정지
        if dist_y < 20: # 20 픽셀 이내면 주차 완료
            self.state = ParkingState.FINISHED
            self.publish_control(0.0, 0.0)
            self.get_logger().info(f"Success! Parking Finished (DistY: {dist_y:.1f}).")
            return

        # [핵심] 무조건 조향 0.0, 후진 -30.0
        # 흔들릴 이유를 원천 차단
        self.publish_control(0.0, -30.0, alpha=1.0)

    def run_finished(self):
        self.publish_control(0.0, 0.0)
        # 최초 1회만 저장하기 위한 플래그 처리 필요 (안 그러면 루프마다 저장됨)
        if not getattr(self, 'evaluation_saved', False):
            self.save_evaluation_data()
            self.evaluation_saved = True # 저장 완료 플래그

        if hasattr(self, 'gui_handle') and self.gui_handle:
            self.gui_handle.root.after(3000, self.gui_handle.close)

    # --- Helper Methods ---
    def get_tracked_target(self):
        """가장 가까운 타겟 반환 (Target Locking 개념 적용)"""
        if self.latest_bev_msg is None or not self.latest_bev_msg.objects: return None
        
        # 화면 중앙 하단(내 차 근처)에 있는 객체를 우선시
        # 이전 프레임 위치가 있다면 그것과 가까운 것 우선
        
        candidates = []
        # 기준점: 이전에 알던 타겟 위치가 있으면 그것, 없으면 내 차 위치(BEV_CENTER)
        if self.locked_target_pos != (0,0):
            ref_x, ref_y = self.locked_target_pos
        else:
            ref_x, ref_y = self.BEV_CENTER_X, self.BEV_CENTER_Y + 200 # 차 뒤쪽

        for obj in self.latest_bev_msg.objects:
            dist = math.hypot(obj.center_bev.x - ref_x, obj.center_bev.y - ref_y)
            candidates.append((dist, obj))
        
        candidates.sort(key=lambda x: x[0])
        
        if not candidates: return None
        best_dist, best_target = candidates[0]
        
        # 너무 멀면(300px 이상) 다른 주차칸으로 간주하고 무시
        if best_dist > 300.0: return None
        
        # 위치 업데이트
        self.locked_target_pos = (best_target.center_bev.x, best_target.center_bev.y)
        return best_target

    def normalize_angle(self, ang_deg: float) -> float:
        """Normalize angle to [-180, 180]."""
        a = (ang_deg + 180.0) % 360.0 - 180.0
        return a

    def publish_control(self, steer, speed, alpha=0.7):
        filtered_steer = (alpha * steer) + ((1 - alpha) * self.prev_steer)
        self.prev_steer = filtered_steer
        msg = PathPlanningResult()
        msg.x_points = [float(filtered_steer)]
        msg.y_points = [float(speed)]
        self.path_pub.publish(msg)

    def odom_callback(self, msg: Odometry):
        # 위치 저장
        self.current_odom['x'] = msg.pose.pose.position.x
        self.current_odom['y'] = msg.pose.pose.position.y
        
        # 쿼터니언 -> 오일러(Yaw) 변환
        q = msg.pose.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        yaw_rad = math.atan2(siny_cosp, cosy_cosp)
        self.current_odom['yaw'] = math.degrees(yaw_rad)

    def save_evaluation_data(self):
        if self.target_spot_id is None: return

        # 1. 시간 계산
        end_time = self.get_clock().now()
        duration = (end_time - self.state_start_time).nanoseconds / 1e9
        
        # 2. 오차 계산 (현재 Odom vs 정답지)
        ideal = self.IDEAL_POSES.get(self.target_spot_id, {'x':0, 'y':0, 'yaw':0})
        
        # Lateral Error: 유클리드 거리 (cm 단위 변환)
        dist_error_m = math.sqrt((self.current_odom['x'] - ideal['x'])**2 + 
                                 (self.current_odom['y'] - ideal['y'])**2)
        lateral_error_cm = dist_error_m * 100.0
        
        # Heading Error: 각도 차이 (절대값)
        yaw_diff = abs(self.current_odom['yaw'] - ideal['yaw'])
        # 각도 차이는 180도를 넘을 수 없음 (예: 350도와 10도 차이는 20도)
        if yaw_diff > 180: yaw_diff = 360 - yaw_diff
        
        # 3. CSV 저장
        filename = f"parking_evaluation_spot_{self.target_spot_id}.csv"
        file_exists = os.path.isfile(filename)
        
        with open(filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            # 헤더 작성 (파일 처음 만들 때만)
            if not file_exists:
                writer.writerow(['Date', 'Target_Spot', 'Duration(s)', 'Lateral_Err(cm)', 'Heading_Err(deg)', 'Final_X', 'Final_Y', 'Final_Yaw'])
            
            # 데이터 작성
            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                self.target_spot_id,
                f"{duration:.2f}",
                f"{lateral_error_cm:.2f}",
                f"{yaw_diff:.2f}",
                f"{self.current_odom['x']:.3f}",
                f"{self.current_odom['y']:.3f}",
                f"{self.current_odom['yaw']:.3f}"
            ])
            
        self.get_logger().info(f"📊 Evaluation Saved to {filename}!")
        self.get_logger().info(f"   Time: {duration:.1f}s, LatErr: {lateral_error_cm:.1f}cm, AngErr: {yaw_diff:.1f}deg")
        self.get_logger().info(f"   Current Pose: x={self.current_odom['x']:.2f}, y={self.current_odom['y']:.2f}, yaw={self.current_odom['yaw']:.2f}")

def main(args=None):
    rclpy.init(args=args)
    node = ParkingPlannerNode2()
    gui = ParkingGUI(node)
    node.gui_handle = gui
    t = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    t.start()
    try: gui.root.mainloop()
    except KeyboardInterrupt: pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()