# BEV(Bird's-Eye View) 상의 단일 객체 정보를 담는 메시지
string class_name # "parking_spot" 또는 "car"

# BEV 이미지 상에서의 위치, 크기, 각도 (픽셀 단위)
interfaces_pkg/Point2D center_bev
float32 width_bev
float32 height_bev
float32 angle_bev
float32 back_line_angle_bev

# --- [추가할 내용] 꼭짓점 정보 ---
geometry_msgs/Point p1_bev
geometry_msgs/Point p2_bev
geometry_msgs/Point p3_bev
geometry_msgs/Point p4_bev
