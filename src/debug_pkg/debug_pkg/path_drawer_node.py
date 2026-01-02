import rclpy
from rclpy.node import Node
import numpy as np
import cv2
from cv_bridge import CvBridge

import tf2_ros
from tf2_ros import TransformException
from rclpy.duration import Duration

from interfaces_pkg.msg import MotionCommand
from sensor_msgs.msg import Image

class PathImageCreatorNode(Node):
    def __init__(self):
        super().__init__('path_image_creator_node')

        self.image_width = self.declare_parameter('image_width', 800).value
        self.image_height = self.declare_parameter('image_height', 800).value
        self.resolution = self.declare_parameter('resolution_m_per_px', 0.05).value
        self.output_path = self.declare_parameter('output_path', '~/parking_path_result.png').value

        # 4채널(BGRA) 투명 캔버스 생성
        self.canvas = np.zeros((self.image_height, self.image_width, 4), dtype=np.uint8)
        self.bridge = CvBridge()
        self.last_pixel_coords = None
        self.is_reversing = False
        
        # 경로의 모든 좌표를 저장할 리스트
        self.path_points = [] 

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.image_publisher = self.create_publisher(Image, '/path_image', 10)
        self.motion_cmd_subscriber = self.create_subscription(
            MotionCommand, '/topic_control_signal', self.motion_cmd_callback, 10)
        self.timer = self.create_timer(0.05, self.timer_callback)

        self.get_logger().info("✅ Path Image Creator Node (Final Version) has started.")
        self.get_logger().info(f"Path image will be saved to '{self.output_path}' on shutdown.")

    def motion_cmd_callback(self, msg: MotionCommand):
        self.is_reversing = msg.left_speed < 0 or msg.right_speed < 0

    def timer_callback(self):
        from_frame = 'odom'
        to_frame = 'chassis'
        try:
            t = self.tf_buffer.lookup_transform(from_frame, to_frame, rclpy.time.Time())
        except TransformException as ex:
            self.get_logger().warn(f'Could not transform {to_frame} to {from_frame}: {ex}')
            return

        world_x = t.transform.translation.x
        world_y = t.transform.translation.y

        center_x_px = self.image_width // 2
        center_y_px = self.image_height // 2
        
        current_x_px = center_x_px + int(world_x / self.resolution)
        current_y_px = center_y_px - int(world_y / self.resolution)
        
        current_pixel_coords = (current_x_px, current_y_px)

        # 제안된 새로운 색상 (BGR + Alpha)
        color = (255, 0, 255, 255) if self.is_reversing else (0, 255, 0, 255)

        if self.last_pixel_coords is not None:
            if self.last_pixel_coords != current_pixel_coords:
                # 안티에일리어싱(lineType=cv2.LINE_AA)을 적용하여 선을 부드럽게 그립니다.
                cv2.line(self.canvas, self.last_pixel_coords, current_pixel_coords, color, thickness=3, lineType=cv2.LINE_AA)
                
                self.path_points.append(current_pixel_coords)

                # 20번째 점마다 화살표를 그려 주행 방향을 표시합니다.
                if len(self.path_points) > 5 and len(self.path_points) % 20 == 0:
                    prev_point = self.path_points[-6]
                    cv2.arrowedLine(self.canvas, prev_point, current_pixel_coords, color, thickness=2, tipLength=0.4, line_type=cv2.LINE_AA)

        else: # 맨 처음 지점일 경우
            self.path_points.append(current_pixel_coords)
            # 시작점 표시 (녹색 원)
            cv2.circle(self.canvas, current_pixel_coords, radius=8, color=(0, 150, 0, 255), thickness=-1, lineType=cv2.LINE_AA)
            cv2.putText(self.canvas, "Start", (current_pixel_coords[0] + 15, current_pixel_coords[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0, 255), 2, cv2.LINE_AA)

        self.last_pixel_coords = current_pixel_coords
        self.image_publisher.publish(self.bridge.cv2_to_imgmsg(self.canvas, "bgra8"))
    
    # path_image_creator_node.py 파일의 save_image 함수를 아래 코드로 교체
    def save_image(self):
        temp_canvas = self.canvas.copy() # 원본 캔버스 보존을 위해 복사본 사용

        # 경로 시작/종료점에 원 그리기 (텍스트는 나중에 회전된 이미지에 그릴 것임)
        if self.path_points:
            # 시작점 원 (녹색)
            start_point = self.path_points[0]
            cv2.circle(temp_canvas, start_point, radius=8, color=(0, 150, 0, 255), thickness=-1, lineType=cv2.LINE_AA)
            
            # 종료점 원 (진한 빨강)
            end_point = self.path_points[-1]
            cv2.circle(temp_canvas, end_point, radius=8, color=(0, 0, 200, 255), thickness=-1, lineType=cv2.LINE_AA)

        # --- 이미지 90도 시계 반대 방향 회전 ---
        # OpenCV는 (width, height) 형식으로 크기를 다루므로, 행렬 변환을 사용
        (h, w) = temp_canvas.shape[:2]
        center = (w // 2, h // 2)
        # 90도 시계 반대 방향 회전 (angle=-90)
        M = cv2.getRotationMatrix2D(center, 90, 1.0) # -90 대신 90도를 넣으면 시계방향 회전
        rotated_canvas = cv2.warpAffine(temp_canvas, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0)) # 배경은 투명하게 유지

        # --- 회전된 이미지에 텍스트 추가 ---
        # 텍스트는 회전된 이미지의 새로운 좌표계에 맞춰 배치해야 함
        
        # 텍스트 위치 계산 (예시: 회전 전 시작점과 끝점을 기준으로)
        # 시작점 텍스트 위치: 회전 전 시작점 (start_point.x, start_point.y)가
        # 회전 후 어디로 갈지 계산해야 합니다.
        # 90도 회전 후 (x_new, y_new) = (y_old, width - x_old) (원점 기준)
        # 여기서는 이미지 중앙 기준이므로 좀 더 복잡하지만, 대략적인 위치를 조정할 수 있습니다.
        
        # 시작점 텍스트 위치 (수동 조정 필요, 경로 위치에 따라 달라짐)
        # 예시: 회전된 이미지의 특정 위치에 텍스트 삽입
        # 이 값들은 경로의 실제 시작/종료 위치와 이미지 크기에 따라 조정이 필요합니다.
        # 아래 예시는 현재 이미지에서 Start/End 위치를 추정하여 작성된 것입니다.
        start_text_pos = (self.image_width // 2 - 10, 100) # 대략 상단 중앙 부근
        end_text_pos = (self.image_width // 2 + 10, 300) # 대략 중간 부근

        # 시작점 텍스트
        cv2.putText(rotated_canvas, "Start", (start_text_pos[0], start_text_pos[1]), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0, 255), 2, cv2.LINE_AA)
        # 종료점 텍스트
        cv2.putText(rotated_canvas, "End", (end_text_pos[0], end_text_pos[1]), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0, 255), 2, cv2.LINE_AA)


        import os
        path = os.path.expanduser(self.output_path)
        try:
            # 회전된 캔버스를 저장
            cv2.imwrite(path, rotated_canvas)
            self.get_logger().info(f"✅ Successfully saved rotated transparent path image to: {path}")
        except Exception as e:
            self.get_logger().error(f"Failed to save image: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = PathImageCreatorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Keyboard Interrupt (SIGINT), shutting down...')
    finally:
        node.save_image()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()