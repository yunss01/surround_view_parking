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
    CALCULATING = 3         
    ARC_1_SWING = 4         
    ARC_2_REVERSE = 5       
    ALIGNMENT = 6           
    FINISHED = 7            

class ParkingGUI:
    def __init__(self, ros_node):
        self.node = ros_node
        self.root = tk.Tk()
        self.root.title("Real-Car Parking Selector")
        self.root.geometry("300x400")

        tk.Label(self.root, text="Select Target Spot", font=("Arial", 14, "bold")).pack(pady=10)

        self.btn_spot_1 = tk.Button(self.root, text="Parking Spot 1", font=("Arial", 12), height=2, width=20,
                                    command=lambda: self.select_spot(1))
        self.btn_spot_1.pack(pady=5)

        self.btn_spot_2 = tk.Button(self.root, text="Parking Spot 2", font=("Arial", 12), height=2, width=20,
                                    command=lambda: self.select_spot(2))
        self.btn_spot_2.pack(pady=5)

        self.btn_spot_3 = tk.Button(self.root, text="Parking Spot 3", font=("Arial", 12), height=2, width=20,
                                    command=lambda: self.select_spot(3))
        self.btn_spot_3.pack(pady=5)

        self.btn_spot_4 = tk.Button(self.root, text="Parking Spot 4", font=("Arial", 12), height=2, width=20,
                                    command=lambda: self.select_spot(4))
        self.btn_spot_4.pack(pady=5)

        self.btn_stop = tk.Button(self.root, text="EMERGENCY STOP", font=("Arial", 12, "bold"), bg="red", fg="white", height=2, width=20,
                                  command=self.emergency_stop)
        self.btn_stop.pack(pady=20)

    def select_spot(self, spot_id):
        self.node.set_target_spot(spot_id)

    def emergency_stop(self):
        self.node.force_stop()

    def run(self):
        self.root.mainloop()

class ParkingPlannerNode2(Node):
    def __init__(self):
        super().__init__('parking_planner_node_2')

        self.BEV_CENTER_X = 850 / 2
        self.BEV_CENTER_Y = 1150 / 2
        self.max_steering = 7.0

        # [실제 차량용 설정] 접근 시간 및 진입 허용 오차
        # 실제 환경에서는 바닥 마찰 등으로 시간이 다를 수 있으니 현장에서 튜닝 필요할 수 있음
        self.approach_duration_map = {1: 9.3, 2: 13.6, 3: 17.6, 4: 21.6}
        self.approach_speed = 50.0 # 모터 제어기에 맞는 속도값
        
        # 2.0도 정밀 진입 조건
        self.arc2_finish_threshold_map = {1: 2.0, 2: 2.0, 3: 2.0, 4: 2.0}

        self.state = ParkingState.READY
        self.target_spot_id = None 
        self.latest_bev_msg = None 
        self.state_start_time = None
        
        self.locked_target_angle = 0.0
        self.locked_target_pos = (0, 0)
        
        self.prev_steer = 0.0
        self.alignment_count = 0
        self.switch_angle_threshold = 30.0
        
        # One-Shot 정렬 플래그
        self.has_corrected_initial_angle = False
        
        # QoS 설정
        self.qos = QoSProfile(reliability=QoSReliabilityPolicy.RELIABLE, history=QoSHistoryPolicy.KEEP_LAST, depth=1)
        
        # [Perception] BEV 결과 구독
        self.bev_sub = self.create_subscription(BevPerceptionResult, SUB_BEV_PERCEPTION_TOPIC, self.bev_callback, self.qos)
        
        # [Control] 경로 및 상태 발행
        self.path_pub = self.create_publisher(PathPlanningResult, PUB_PARKING_PATH_TOPIC, self.qos)
        self.status_pub = self.create_publisher(String, PUB_PARKING_STATUS_TOPIC, self.qos)

        # [제거됨] Odom 구독 및 관련 변수 삭제

        self.start_arc1_angle = None
        
        # GUI 쓰레드 시작
        self.gui = ParkingGUI(self)
        self.gui_thread = threading.Thread(target=self.gui.run)
        self.gui_thread.daemon = True
        self.gui_thread.start()

        self.timer = self.create_timer(0.05, self.control_loop)
        self.get_logger().info("✅ Real-Car Parking Planner Ready. Waiting for GUI input...")

    def bev_callback(self, msg: BevPerceptionResult):
        self.latest_bev_msg = msg

    def set_target_spot(self, idx):
        self.target_spot_id = idx
        self.state_start_time = self.get_clock().now()
        self.alignment_count = 0
        self.has_corrected_initial_angle = False 
        self.state = ParkingState.APPROACHING
        self.get_logger().info(f"🚀 Command Received: Start Parking into Spot {idx}")

    def force_stop(self):
        self.state = ParkingState.READY
        self.publish_control(0.0, 0.0)
        self.get_logger().warn("🚨 EMERGENCY STOP TRIGGERED")

    def calculate_entrance_angle(self, obj):
        dx = obj.p2_bev.x - obj.p1_bev.x
        dy = obj.p2_bev.y - obj.p1_bev.y
        return math.degrees(math.atan2(dy, dx))

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

    def run_approaching(self):
        elapsed = (self.get_clock().now() - self.state_start_time).nanoseconds / 1e9
        duration = self.approach_duration_map.get(self.target_spot_id, 5.0)
        if elapsed < duration: self.publish_control(0.0, self.approach_speed)
        else:
            self.publish_control(0.0, 0.0)
            time.sleep(0.5)
            self.state = ParkingState.SEARCHING

    def run_searching(self):
        if self.latest_bev_msg is None or not self.latest_bev_msg.objects: return
        try:
            target_obj = max(self.latest_bev_msg.objects, key=lambda o: o.center_bev.y)
            self.locked_target_angle = self.calculate_entrance_angle(target_obj)
            self.locked_target_pos = (target_obj.center_bev.x, target_obj.center_bev.y)
            self.get_logger().info(f"Spot Found. Angle: {self.locked_target_angle:.2f}")
            self.state = ParkingState.CALCULATING
        except ValueError: return

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

    # --- 5. 제2 원호 (Blind Straight + 2.0 deg Threshold) ---
    def run_arc_2_reverse(self):
        target_obj = self.get_tracked_target()
        
        # [Safety] 타겟 놓치면 회전 중지하고 직진 후진
        if target_obj is None:
            self.publish_control(0.0, -10.0, alpha=1.0) 
            return

        current_angle = self.calculate_entrance_angle(target_obj)
        target_threshold = self.arc2_finish_threshold_map.get(self.target_spot_id, 3.0)
        
        if abs(current_angle) < target_threshold: self.alignment_count += 1
        else: self.alignment_count = 0

        if self.alignment_count >= 3:
            self.prev_steer = 0.0 
            self.publish_control(0.0, 0.0, alpha=1.0) # 즉시 정지
            time.sleep(1.0) 
            self.get_logger().info(f"Arc 2 Finished. Aligned ({current_angle:.1f}). Force Straight Mode.")
            self.state = ParkingState.ALIGNMENT
            self.has_corrected_initial_angle = False 
            return

        # [Slow Approach] -10.0 속도
        self.publish_control(-self.max_steering, -10.0, alpha=0.3)

    # --- 6. 정렬 (One-Shot Correction + Straight) ---
    def run_alignment(self):
        target_obj = self.get_tracked_target()
        
        if target_obj is None:
             self.publish_control(0.0, -10.0, alpha=1.0)
             return

        dist_y = abs(target_obj.center_bev.y - self.BEV_CENTER_Y)

        # [종료 조건] 실제 차량에선 거리가 다를 수 있으니 20픽셀이 적당한지 현장 확인 필요
        if dist_y < 20: 
            self.state = ParkingState.FINISHED
            self.publish_control(0.0, 0.0)
            self.get_logger().info(f"Success! Parking Finished (DistY: {dist_y:.1f}).")
            return

        # [One-Shot] 초기 1회 보정 로직
        if not self.has_corrected_initial_angle:
            current_angle = self.calculate_entrance_angle(target_obj)
            
            if abs(current_angle) > 1.0:
                correction = -current_angle * 0.5 
                correction = max(min(correction, 3.0), -3.0)
                
                self.publish_control(correction, -10.0, alpha=1.0)
                self.get_logger().info(f"[One-Shot] Correcting Initial Angle {current_angle:.1f} -> Steer {correction:.1f}")
                return 
            else:
                self.has_corrected_initial_angle = True
                self.get_logger().info("[One-Shot] Correction Done. Locking to 0.0")

        # [기본] 직진 모드
        self.publish_control(0.0, -10.0, alpha=1.0)

    def run_finished(self):
        # 완료되면 정지하고 대기
        self.publish_control(0.0, 0.0)
        # CSV 저장 로직 삭제됨
        # 상태를 READY로 돌릴지, 아니면 FINISHED에서 멈춰있을지 결정 (여기선 멈춤 유지)
        if self.alignment_count == 0: # 로그 한 번만 찍기 위한 꼼수
             self.get_logger().info("🅿️ Parking Complete. Waiting for new command.")
             self.alignment_count = 1

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
        
        now = self.get_clock().now().nanoseconds / 1e9
        if not hasattr(self, 'last_log_time'): self.last_log_time = 0
        if now - self.last_log_time > 1.0:
            self.get_logger().info(f"[Control] Cmd: {filtered_steer:.2f} | State: {self.state.name}")
            self.last_log_time = now

        msg = PathPlanningResult()
        msg.x_points = [float(filtered_steer)]
        msg.y_points = [float(speed)]
        self.path_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = ParkingPlannerNode2()
    try:
        rclpy.spin(node)
    except SystemExit:
        pass 
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()