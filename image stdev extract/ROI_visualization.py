import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Rectangle
import os

# --- [1. 스타일 및 폰트 설정 (Booktabs 스타일)] ---
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11
})

# 1. 이미지 로드 (회원님 제안대로 2.png 사용)
image_path = '2.png'
if not os.path.exists(image_path):
    print(f"❌ Error: Cannot find image file at {image_path}")
    exit()

img = mpimg.imread(image_path)
h, w, _ = img.shape
print(f"Image Resolution: {w}x{h}")


# 2. sim_parking_planner_node.py의 최신 좌표 정의
# [공통 X 범위 (Lateral)]
roi_x1, roi_x2 = 700, 850
roi_width = roi_x2 - roi_x1

# [Y 좌표 범위 (Longitudinal) - Gazebo 좌표계 방향 반영]
# A. 주차 실패 점유 확인 영역 (Target Spot)
occ_y1, occ_y2 = 550, 650
occ_height = occ_y2 - occ_y1

# B. 인접 장애물 탐지 영역 (Adjacent Spots)
# 노드 코드의 left_y, right_y 좌표를 그대로 사용
adj_1_y1, adj_1_y2 = max(0, 850), max(0, 950)   # 1번 칸 (위)
adj_3_y1, adj_3_y2 = min(h, 350), min(h, 450)   # 3번 칸 (아래)


# 3. 그래프 그리기 (논문 단일 컬럼 폭에 적합한 figsize)
fig, ax = plt.subplots(figsize=(8, 6))

# 백그라운드 이미지 표시
ax.imshow(img, cmap='gray' if img.ndim==2 else None)


# --- [ROI 박스 그리기] ---
# Matplotlib의 Rectangle은 (x, y), width, height 순서임. 
# y 좌표는 이미지의 상단이 0이므로, 작은 값을 y로 설정.
thickness = 3

# A. Occupancy Check -> RED
rect_occ = Rectangle((roi_x1, occ_y1), roi_width, occ_height, 
                     linewidth=thickness, edgecolor='red', facecolor='none', label='Occupancy Check (Target Spot)')
ax.add_patch(rect_occ)


# B. Adjacent Obstacle Checks -> BLUE
# 박스가 여러 개라도 범례에는 하나만 뜨게 하려면 label을 한 번만 지정
# Obs 1 (위)
rect_obs1 = Rectangle((roi_x1, adj_1_y1), roi_width, (adj_1_y2 - adj_1_y1), 
                      linewidth=thickness, edgecolor='royalblue', facecolor='none', label='Adjacent Obstacle Check')
ax.add_patch(rect_obs1)

# Obs 3 (아래) - label은 생략
rect_obs3 = Rectangle((roi_x1, adj_3_y1), roi_width, (adj_3_y2 - adj_3_y1), 
                      linewidth=thickness, edgecolor='royalblue', facecolor='none')
ax.add_patch(rect_obs3)


# --- [그래프 디테일 설정] ---
# 불필요한 테두리와 눈금 제거
ax.axis('off')

# 범례를 이미지 내부 우상단에 배치 (하얀 배경)
handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))

# frameon=True, framealpha=0.9 로 설정하여 배경 이미지가 글자를 가리지 않게 흰색 네모칸 생성
ax.legend(by_label.values(), by_label.keys(), loc='upper right', 
          fontsize=12, frameon=True, framealpha=0.9, edgecolor='black')

plt.tight_layout()

# --- [NEW: PNG와 PDF 동시 저장] ---
output_path_png = 'SVM_ROI_Visualization_LaTexStyle.png'
output_path_pdf = 'SVM_ROI_Visualization_LaTexStyle.pdf'

plt.savefig(output_path_png, dpi=300, bbox_inches='tight', pad_inches=0.1)
plt.savefig(output_path_pdf, format='pdf', bbox_inches='tight', pad_inches=0.1)

print(f"✅ Visualization saved to {output_path_png} and {output_path_pdf}")

plt.show()
