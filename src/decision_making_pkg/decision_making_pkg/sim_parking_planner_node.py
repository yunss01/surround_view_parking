import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSDurabilityPolicy, QoSReliabilityPolicy
from enum import Enum
import numpy as np
import math
import time
import threading
import tkinter as tk # GUI용 라이브러리
from geometry_msgs.msg import PoseStamped 

# 사용할 메시지 타입
from interfaces_pkg.msg import BevPerceptionResult, PathPlanningResult, BevObject
from std_msgs.msg import String
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped

#---------------Variable Setting---------------
SUB_BEV_PERCEPTION_TOPIC = "bev_perception_result"

PUB_PARKING_PATH_TOPIC = "parking_path"
PUB_PARKING_STATUS_TOPIC = "parking_status"
#----------------------------------------------

class ParkingState(Enum):
    READY = 0               
    APPROACHING = 1         # 해당 주차칸 근처로 이동 (Blind)
    SEARCHING = 2           # 멈춰서 타겟 주차 공간 탐색 (경로 생성 X)
    SWING_OUT=3             # 좌회전 전진으로 각도 확보
    REVERSE_ENTRY = 4       # [Step 1] 최대 조향으로 엉덩이 집어넣기 (Turn-in)
    ALIGNMENT = 5           # [Step 2] 비전 기반 정렬 및 진입 (Visual Servoing)
    FINISHED = 6            # 주차 완료

class ParkingGUI:
    def __init__(self, ros_node):
        self.node = ros_node
        self.root = tk.Tk()
        self.root.title("Heuristic Parking")
        self.root.geometry("300x300")
        tk.Label(self.root, text="Select Target Spot", font=("Arial", 14, "bold")).pack(pady=10)
        self.spot_buttons = {}
        for i in range(1, 5):
            btn = tk.Button(self.root, text=f"Park at Spot {i}", 
                            font=("Arial", 12), height=2, width=20, bg="white",
                            command=lambda idx=i: self.select_spot(idx))
            btn.pack(pady=5)
            self.spot_buttons[i] = btn

    def close(self):
        self.root.destroy()
        self.node.get_logger().info("GUI Closed.")

    def select_spot(self, idx):
        self.node.get_logger().info(f"GUI: Target Spot {idx} Selected!")
        self.node.set_target_spot(idx)

        # 버튼 색상 즉시 업데이트
        for i, btn in self.spot_buttons.items():
            if i == idx:
                btn.config(bg="#4CAF50", fg="white") # Green
            else:
                btn.config(bg="white", fg="black")
        
        self.root.update_idletasks()
        self.root.update()

class ParkingPlannerNode(Node):
    def __init__(self):
        super().__init__('parking_planner_node')

        self.BEV_CENTER_X = 850 / 2
        self.BEV_CENTER_Y = 1150 / 2
        self.max_steering = 7.0

        # === 하드코딩된 접근 시간 (초) ===
        # 각 주차칸 앞까지 가는 시간 (속도 50 기준)
        self.approach_duration_map = {
            1: 11.2, 2: 15.2, 3: 19.2, 4: 23.2
        }
        self.approach_speed = 50.0

        self.state = ParkingState.READY
        self.target_spot_id = None
        self.latest_bev_msg = None 
        self.state_start_time = None
        self.tracked_target_center = None # 타겟 추적을 위한 변수 (마지막으로 본 타겟의 위치)
        self.blind_counter = 0 # 타겟 놓침 방지 카운터
        self.prev_steer = 0.0
        self.last_known_angle = 0.0
        self.smooth_lat_error = 0.0 # 정밀 주차를 위한 데이터 스무딩 변수
        self.smooth_heading_error = 0.0
         
        self.qos = QoSProfile(reliability=QoSReliabilityPolicy.RELIABLE, history=QoSHistoryPolicy.KEEP_LAST, depth=1)
        self.bev_sub = self.create_subscription(BevPerceptionResult, SUB_BEV_PERCEPTION_TOPIC, self.bev_callback, self.qos)
        
        self.path_pub = self.create_publisher(PathPlanningResult, PUB_PARKING_PATH_TOPIC, self.qos)
        self.status_pub = self.create_publisher(String, PUB_PARKING_STATUS_TOPIC, self.qos)

        # Main Loop (0.1s)
        self.timer = self.create_timer(0.1, self.control_loop)
        
        self.get_logger().info("✅ Heuristic Parking Planner Started.")

    def bev_callback(self, msg: BevPerceptionResult):
        self.latest_bev_msg = msg

    def set_target_spot(self, idx):
        self.target_spot_id = idx
        self.state_start_time = self.get_clock().now()
        
        # 변수 초기화 (중요!)
        self.tracked_target_center = None 
        self.entry_start_time = None
        self.prev_steer = 0.0
        self.last_known_angle = 0.0
        self.blind_counter = 0
        self.smooth_lat_error = 0.0
        self.smooth_heading_error = 0.0
        self.state = ParkingState.APPROACHING
        self.get_logger().info(f"Start Approaching Spot {idx}...")

    def force_stop(self):
        self.state = ParkingState.READY
        self.publish_control(0.0, 0.0)

    def calculate_entrance_angle(self, obj):
        # Extractor가 입구 점을 p1, p2에 넣어주므로 바로 사용
        dx = obj.p2_bev.x - obj.p1_bev.x
        dy = obj.p2_bev.y - obj.p1_bev.y
            
        angle_rad = math.atan2(dy, dx)
        angle_deg = math.degrees(angle_rad)
        
        # 0도 = 수평, 90도 = 수직
        angle_deg = abs(angle_deg) % 180
        if angle_deg > 90:
            angle_deg = 180 - angle_deg
            
        return angle_deg

    # --- Main Loop ---
    def control_loop(self):
        # 상태별 로직 실행
        if self.state == ParkingState.READY:
            pass 

        elif self.state == ParkingState.APPROACHING:
            self.run_approaching()

        elif self.state == ParkingState.SEARCHING:
            self.run_searching()

        elif self.state == ParkingState.SWING_OUT:
            self.run_swing_out()

        elif self.state == ParkingState.REVERSE_ENTRY:
            self.run_reverse_entry()

        elif self.state == ParkingState.ALIGNMENT:
            self.run_alignment()

        elif self.state == ParkingState.FINISHED:
            self.run_finished()
            
        self.status_pub.publish(String(data=self.state.name))

    # --- State Logic ---

    def run_approaching(self):
        """1. 해당 주차칸 근처까지 직진 (Open-Loop)"""
        elapsed = (self.get_clock().now() - self.state_start_time).nanoseconds / 1e9
        duration = self.approach_duration_map.get(self.target_spot_id, 5.0)

        if elapsed < duration:
            self.publish_control(0.0, self.approach_speed)
        else:
            self.publish_control(0.0, 0.0)
            time.sleep(0.5) 
            self.get_logger().info("Approach Done. Switching to SEARCHING...")
            self.state = ParkingState.SEARCHING

    def run_searching(self):
        """2. 멈춰서 주차 공간이 보이는지 확인"""
        if self.latest_bev_msg is None or not self.latest_bev_msg.objects:
            self.get_logger().warn("Searching: No parking spot detected!", throttle_duration_sec=1)
            # 안 보이면 살짝 더 전진하거나 후진하는 로직 추가 가능
            return 

        # 가장 가까운(화면 하단, Y값이 큰) 주차 공간 확인
        try:
            target_obj = max(self.latest_bev_msg.objects, key=lambda o: o.center_bev.y)
            self.tracked_target_center = (target_obj.center_bev.x, target_obj.center_bev.y)
            angle = self.calculate_entrance_angle(target_obj)

            self.get_logger().info(f"Spot Found! Angle: {angle:.2f}. Starting SWING_OUT.")
            self.state = ParkingState.SWING_OUT
        except ValueError:
            return

    def get_tracked_target(self):
        """저장된 위치와 가장 가까운 객체를 찾아 반환 (Tracking)"""
        if self.latest_bev_msg is None or not self.latest_bev_msg.objects:
            return None
        
        if self.tracked_target_center is None:
            return None

        # 이전에 알던 위치와 가장 가까운 객체 찾기
        last_x, last_y = self.tracked_target_center
        # 1. 모든 객체와의 거리 계산 및 정렬
        # (거리, 객체) 튜플 리스트 생성
        candidates = []
        for obj in self.latest_bev_msg.objects:
            dist = math.hypot(obj.center_bev.x - last_x, obj.center_bev.y - last_y)
            candidates.append((dist, obj))
        
        # 거리순 오름차순 정렬
        candidates.sort(key=lambda x: x[0])
        
        if not candidates:
            return None

        best_dist, best_target = candidates[0]

        # [핵심 수정] Distance Gating
        # 만약 가장 가까운 객체조차 300픽셀 이상 떨어져 있다면, 이는 내가 쫓던 타겟이 아님 (옆 칸임)
        if best_dist > 300.0:
            # 타겟을 놓친 것으로 간주
            return None

        # 위치 업데이트
        self.tracked_target_center = (best_target.center_bev.x, best_target.center_bev.y)
        return best_target

    def run_swing_out(self):
        """[Step 2.5] 좌회전 전진하여 진입 각도 확보"""
        target_obj = self.get_tracked_target()

        if target_obj is None:
            self.blind_counter += 1
            # 20프레임(약 2초) 동안은 안 보여도 그냥 하던 대로 가라!
            if self.blind_counter < 20:
                self.publish_control(-self.max_steering, 50.0) # 좌회전 전진 유지
                self.get_logger().warn(f"Blind Swing ({self.blind_counter}/20)... Keeping turn.", throttle_duration_sec=0.5)
                return
            else:
                self.publish_control(0.0, 0.0)
                self.get_logger().error("Lost Target in Swing-out! Stopping.")
                return

        # 타겟을 찾았으면 카운터 초기화
        self.blind_counter = 0
        
        current_angle = self.calculate_entrance_angle(target_obj)
        
        # 75도 이하가 되면 다음 단계로
        if current_angle < 75.0:
            self.get_logger().info(f"Swing-out Complete (Angle={current_angle:.2f}). Switching to REVERSE_ENTRY.")
            self.publish_control(0.0, 0.0)
            time.sleep(0.5)
            self.state = ParkingState.REVERSE_ENTRY
            return
        
        steer = -self.max_steering
        speed = 50.0
        
        self.publish_control(steer, speed)
        self.get_logger().info(f"Swinging Out: Angle={current_angle:.1f} -> Target < 60.0")

    def run_reverse_entry(self):
        """3. [Turn-in] 최대 조향으로 우회전 후진"""
        target_obj = self.get_tracked_target()

        if target_obj is None:
            self.blind_counter += 1
            if self.blind_counter < 20:
                self.publish_control(-self.max_steering, -50.0) # 우회전 후진 유지
                self.get_logger().warn(f"Blind Reverse ({self.blind_counter}/20)... Trusting trajectory.")
                return

            # 2초 넘게 안 보이면 그때 정지
            if self.last_known_angle < 20.0:
                self.state = ParkingState.ALIGNMENT
                return
            else:
                self.get_logger().error(f"Lost Target too long! Stopping.")
                self.publish_control(0.0, 0.0)
                return

        self.blind_counter = 0 # 타겟 찾으면 카운터 리셋
        # 입구 라인 기반 각도 계산
        current_angle = self.calculate_entrance_angle(target_obj)
        self.last_known_angle = current_angle
        
        # [종료 조건] 입구 라인이 수평(0도)에 가까워지면 = 차가 정렬되면
        # 주의: calculate_entrance_angle의 반환값이 수평선 기준 각도라면, 0도가 평행임.
        if abs(current_angle) < 10.0:
            self.get_logger().info(f"Aligned roughly ({current_angle:.2f} deg). Switching to ALIGNMENT.")
            
            # [중요] 상태 전환 시 일단 정지! (조향 풀 시간을 줌)
            self.publish_control(0.0, 0.0)
            time.sleep(0.5) 
            
            # ALIGNMENT 진입 전 스무딩 변수 초기화
            self.smooth_lat_error = 0.0
            self.smooth_heading_error = 0.0

            self.state = ParkingState.ALIGNMENT
            return

        # 무조건 우회전(Right) 최대 조향
        # 시뮬레이터 최대값(-7.0)을 그대로 사용
        self.publish_control(-self.max_steering, -50.0)
        self.get_logger().info(f"Turning: Angle={current_angle:.1f}")

    def run_alignment(self):
        """4. [Visual Servoing] 정밀 진입"""
        target_obj = self.get_tracked_target()

        if target_obj is None:
            # Alignment 중에는 타겟을 놓쳐도 멈추지 않고 아주 천천히 직진(조향 유지)
            self.publish_control(self.prev_steer, -50.0) 
            self.get_logger().warn("Target Lost temporarily. Creeping back...", throttle_duration_sec=1)
            return

        # 1. Raw Error 계산
        raw_lat_error = self.BEV_CENTER_X - target_obj.center_bev.x
        raw_heading_error = self.calculate_entrance_angle(target_obj)
        dist_y_pixel = abs(target_obj.center_bev.y - self.BEV_CENTER_Y)

        # 오차가 3도 미만이면 그냥 잘 정렬된 것으로 치고 무시해버림
        # 모델이 점을 조금 삐끗하게 찍어서 생긴 노이즈를 여기서 삭제함
        if abs(raw_heading_error) < 3.0:
            raw_heading_error = 0.0

        # 2. [핵심] 입력값 스무딩 (Input Exponential Moving Average)
        # alpha값이 작을수록 노이즈가 줄지만 반응이 느려짐. 0.3~0.5 추천
        input_alpha = 0.4
        
        if self.smooth_lat_error == 0.0: # 첫 진입 시 초기화
            self.smooth_lat_error = raw_lat_error
            self.smooth_heading_error = raw_heading_error
        else:
            self.smooth_lat_error = (input_alpha * raw_lat_error) + ((1 - input_alpha) * self.smooth_lat_error)
            self.smooth_heading_error = (input_alpha * raw_heading_error) + ((1 - input_alpha) * self.smooth_heading_error)

        # 3. 종료 조건 확인
        if dist_y_pixel < 15:
            self.get_logger().info(f"Target Reached! Dist={dist_y_pixel:.1f}. FINISHED.")
            self.state = ParkingState.FINISHED
            return

        # 4. PID 제어 튜닝
        Kp_lat = 0.15 
        Kp_ang = 0.3 
        
        # Deadband (오차가 작으면 조향 0)
        if abs(self.smooth_heading_error) < 2.0 and abs(self.smooth_lat_error) < 15.0:
            steer = 0.0
            log_msg = "STRAIGHT"
        else:
            raw_steer = -((Kp_lat * self.smooth_lat_error) + (Kp_ang * self.smooth_heading_error))
            steer = np.clip(raw_steer, -self.max_steering, self.max_steering)
            log_msg = "Correction"
        
        speed = -50.0

        # 출력 시 LPF를 강하게 적용 (alpha를 낮춤)
        self.publish_control(steer, speed, alpha=0.2)
        
        self.get_logger().info(f"Aligning ({log_msg}): Dist={dist_y_pixel:.0f}, ErrX={self.smooth_lat_error:.1f}, Ang={self.smooth_heading_error:.1f} -> Steer={steer:.1f}")

    def run_finished(self):
        self.publish_control(0.0, 0.0)
        self.get_logger().info("Parking Finished! Shutting down in 3 seconds...", throttle_duration_sec=5)
        if hasattr(self, 'gui_handle') and self.gui_handle:
            self.gui_handle.root.after(3000, self.gui_handle.close)

    def publish_control(self, steer, speed, alpha=0.7):
        # Low Pass Filter (LPF) 적용 - 떨림 방지
        # alpha가 작을수록 부드럽지만 반응이 느림. 0.7 정도가 적당.
        filtered_steer = (alpha * steer) + ((1 - alpha) * self.prev_steer)
        self.prev_steer = filtered_steer

        msg = PathPlanningResult()
        msg.x_points = [float(filtered_steer)]
        msg.y_points = [float(speed)]
        self.path_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    
    node = ParkingPlannerNode()
    gui = ParkingGUI(node)
    node.gui_handle = gui
    
    t = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    t.start()
    
    try:
        gui.root.mainloop()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()