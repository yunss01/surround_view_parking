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
        self.approach_duration_map = {1: 10.5, 2: 14.5, 3: 18.5, 4: 22.5}
        self.approach_speed = 50.0

        self.state = ParkingState.READY
        self.target_spot_id = None
        self.latest_bev_msg = None 
        self.state_start_time = None
        
        # [Target Locking Variables]
        self.locked_target_angle = 0.0  # 고정된 타겟 각도
        self.locked_target_pos = (0, 0) # 고정된 타겟 위치 (상대좌표 아님, 초기 기준)
        self.is_path_calculated = False
        
        # [Dual Smoothing Variables]
        self.smooth_lat_error = 0.0
        self.smooth_heading_error = 0.0
        self.prev_steer = 0.0
        
        # [Two-Arc Params]
        self.switch_angle_threshold = 45.0 # 1차 회전 목표 각도 (Swing Out 각도)
        
        self.qos = QoSProfile(reliability=QoSReliabilityPolicy.RELIABLE, history=QoSHistoryPolicy.KEEP_LAST, depth=1)
        self.bev_sub = self.create_subscription(BevPerceptionResult, SUB_BEV_PERCEPTION_TOPIC, self.bev_callback, self.qos)
        self.path_pub = self.create_publisher(PathPlanningResult, PUB_PARKING_PATH_TOPIC, self.qos)
        self.status_pub = self.create_publisher(String, PUB_PARKING_STATUS_TOPIC, self.qos)

        self.timer = self.create_timer(0.1, self.control_loop)
        self.get_logger().info("✅ Two-Arc Parking Planner (Node 2) Started.")

    def bev_callback(self, msg: BevPerceptionResult):
        self.latest_bev_msg = msg

    def set_target_spot(self, idx):
        self.target_spot_id = idx
        self.state_start_time = self.get_clock().now()
        self.is_path_calculated = False
        self.smooth_lat_error = 0.0
        self.smooth_heading_error = 0.0
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
        angle_deg = abs(angle_deg) % 180
        if angle_deg > 90: angle_deg = 180 - angle_deg
        
        # [보정] 거의 수평이면 0으로 간주 (Jitter 방지)
        if angle_deg < 3.0: return 0.0
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
        # (구현 단순화) 시뮬레이션 환경에 맞춰 45도 각도 확보를 목표로 설정
        # 실제 논문 수식은 R_min 등을 알아야 하므로, 여기서는 Heuristic하게 45도로 고정
        self.switch_angle_threshold = 45.0 
        
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
        
        # 목표 각도(45도 이상)가 되면 정지 및 전환
        # 주의: 각도 정의에 따라 90도에서 줄어드는지, 0도에서 늘어나는지 확인 필요
        # (기존 코드 기준: 90도(수직) -> 0도(수평)으로 변함. Swing Out하면 각도가 줄어듦?)
        # 확인: Swing Out(좌회전)하면 차가 기울어지므로 입구 라인 각도는 변함.
        # 실험적으로 60도 이하가 되면 충분히 꺾은 것으로 판단 (기존 로직 유지)
        
        if current_angle < 60.0: # 45~60도 사이면 충분
            self.publish_control(0.0, 0.0)
            time.sleep(0.5)
            self.get_logger().info(f"Arc 1 Finished. Angle={current_angle:.1f}. Switching to Reverse.")
            self.state = ParkingState.ARC_2_REVERSE
            return

        self.publish_control(-self.max_steering, 50.0) # 좌회전 전진

    # --- 5. 제2 원호 (Reverse Entry) ---
    def run_arc_2_reverse(self):
        """우회전 후진으로 진입 (Turn-in)"""
        target_obj = self.get_tracked_target()

        if target_obj is None:
            self.publish_control(-self.max_steering, -50.0) # 궤적 유지
            return

        current_angle = self.calculate_entrance_angle(target_obj)
        
        # [핵심] Two-Arc 이론상, 이 단계가 끝나면 차는 주차칸과 평행(0도)해야 함
        if abs(current_angle) < 10.0: # 거의 평행해지면
            self.publish_control(0.0, 0.0)
            time.sleep(0.5)
            self.smooth_lat_error = 0.0
            self.smooth_heading_error = 0.0
            self.get_logger().info(f"Arc 2 Finished. Aligned ({current_angle:.1f}). Fine Tuning...")
            self.state = ParkingState.ALIGNMENT
            return

        self.publish_control(-self.max_steering, -50.0) # 우회전 후진

    # --- 6. 정렬 (Hybrid Dual Smoothing) ---
    def run_alignment(self):
        """위치는 민감하게, 각도는 둔감하게 보정"""
        target_obj = self.get_tracked_target()

        if target_obj is None:
            self.publish_control(self.prev_steer, -40.0)
            return

        # 에러 계산
        raw_lat_error = target_obj.center_bev.x - self.BEV_CENTER_X
        raw_heading_error = self.calculate_entrance_angle(target_obj)
        dist_y = abs(target_obj.center_bev.y - self.BEV_CENTER_Y)

        # [Dual Smoothing]
        # 위치(Lat): alpha 0.4 (민감)
        # 각도(Ang): alpha 0.1 (둔감 - Jitter 무시)
        lat_alpha = 0.4
        head_alpha = 0.1
        
        if self.smooth_lat_error == 0.0: self.smooth_lat_error = raw_lat_error
        else: self.smooth_lat_error = (lat_alpha * raw_lat_error) + ((1-lat_alpha)*self.smooth_lat_error)
        
        if self.smooth_heading_error == 0.0: self.smooth_heading_error = raw_heading_error
        else: self.smooth_heading_error = (head_alpha * raw_heading_error) + ((1-head_alpha)*self.smooth_heading_error)

        # 종료 조건
        if dist_y < 20:
            self.state = ParkingState.FINISHED
            return

        # PID 제어 (Deadband 없음)
        Kp_lat = 0.15
        Kp_ang = 0.4
        
        # 위치가 많이 틀어졌으면 각도 무시하고 위치부터
        if abs(self.smooth_lat_error) > 50: Kp_ang = 0.1

        raw_steer = -((Kp_lat * self.smooth_lat_error) + (Kp_ang * self.smooth_heading_error))
        steer = np.clip(raw_steer, -self.max_steering, self.max_steering)
        
        # 속도 고정 (정확성 위함)
        speed = -40.0
        
        self.publish_control(steer, speed, alpha=0.2)
        self.get_logger().info(f"Aligning: X_err={self.smooth_lat_error:.1f}, Ang_err={self.smooth_heading_error:.1f} -> Steer={steer:.1f}")

    def run_finished(self):
        self.publish_control(0.0, 0.0)
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

    def publish_control(self, steer, speed, alpha=0.7):
        filtered_steer = (alpha * steer) + ((1 - alpha) * self.prev_steer)
        self.prev_steer = filtered_steer
        msg = PathPlanningResult()
        msg.x_points = [float(filtered_steer)]
        msg.y_points = [float(speed)]
        self.path_pub.publish(msg)

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