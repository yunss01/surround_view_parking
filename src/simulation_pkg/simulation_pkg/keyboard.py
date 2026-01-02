import rclpy
from rclpy.node import Node
from interfaces_pkg.msg import MotionCommand
import sys, tty, termios

class KeyboardTeleop(Node):
    def __init__(self):
        super().__init__('keyboard_teleop')
        self.publisher = self.create_publisher(MotionCommand, 'topic_control_signal', 10)
        self.speed = 50  # 기본 속도
        self.steering = 0  # 기본 조향

        self.get_logger().info('키보드로 조종하세요: w/s: 전진/후진, a/d: 조향 좌/우, space: 정지, q: 종료')
        self.run()

    def get_key(self):
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch

    def run(self):
        msg = MotionCommand()

        while rclpy.ok():
            key = self.get_key()

            if key == 'w':
                self.speed += 20
            elif key == 's':
                self.speed -= 20
            elif key == 'a':
                self.steering -= 1
            elif key == 'd':
                self.steering += 1
            elif key == ' ':
                self.speed = 0
                self.steering = 0
            elif key == 'q':
                break

            # 속도 제한
            self.speed = max(min(self.speed, 255), -255)
            self.steering = max(min(self.steering, 7), -7)

            msg.left_speed = int(self.speed)
            msg.right_speed = int(self.speed)
            msg.steering = int(self.steering)
            # 후진 시 조향 반전
            if self.speed < 0:
                msg.steering = -msg.steering
                
            self.publisher.publish(msg)
            self.get_logger().info(f'Speed: {self.speed}, Steering: {self.steering}')

def main():
    rclpy.init()
    node = KeyboardTeleop()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
