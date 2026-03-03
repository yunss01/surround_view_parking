import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# 1. 데이터 로드
df = pd.read_csv('parking_evaluation_obstacle1&3_spot_2.csv')

# 정답 좌표 및 규격
IDEAL_POSES = {
    1: {'x': -3.203, 'y': 8.659},
    2: {'x': -3.203, 'y': 5.252},
    3: {'x': -3.203, 'y': 1.883},
    4: {'x': -3.203, 'y': -1.452}
}
target_spot_id = int(df['Target_Spot'].iloc[0])
target_x = IDEAL_POSES[target_spot_id]['x']
target_y = IDEAL_POSES[target_spot_id]['y']

# 그래프 폰트 및 사이즈 설정
plt.rcParams.update({'font.size': 12})
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# --- [Plot 1] X, Y가 반전된 산점도 (검은색 박스 삭제) ---
ax1 = axes[0]

# 15cm 성공 기준 원(빨간 점선) 그리기 (X, Y 교환)
threshold_circle = Circle((target_y, target_x), 0.15, linewidth=2, edgecolor='red', 
                          facecolor='none', linestyle='dashed', label='Success Threshold (15cm)')
ax1.add_patch(threshold_circle)

# 산점도 및 타겟 마커 (X, Y 교환)
ax1.scatter(df['Final_Y'], df['Final_X'], c='royalblue', alpha=0.6, s=40, edgecolors='k', label='Final Position')
ax1.plot(target_y, target_x, 'rX', markersize=12, linewidth=2, label='Target Center')

# 축 범위 설정 (X, Y 범위 교환)
ax1.set_xlim(target_y - 1.0, target_y + 1.0)
ax1.set_ylim(target_x - 0.6, target_x + 0.6)

# 축 라벨 설정 (X, Y 이름 교환)
ax1.set_xlabel('Y Coordinate (m)')
ax1.set_ylabel('X Coordinate (m)')
ax1.set_title(f'Final Positions (Spot {target_spot_id})')
ax1.set_aspect('equal', adjustable='box') 
ax1.legend(loc='upper right')
ax1.grid(True, linestyle=':', alpha=0.7)

# --- [Plot 2] 완벽하게 고정된 각도 오차 히스토그램 ---
ax2 = axes[1]
valid_headings = df.loc[df['Heading_Err(deg)'] < 20, 'Heading_Err(deg)']

ax2.hist(valid_headings, bins=15, color='seagreen', alpha=0.7, edgecolor='black')

ax2.axvline(5.0, color='red', linestyle='dashed', linewidth=2, label='Success Threshold (5 deg)')

ax2.set_xlabel('Heading Error (deg)')
ax2.set_ylabel('Frequency')
ax2.set_title('Distribution of Heading Errors')
ax2.legend()
ax2.grid(True, linestyle=':', alpha=0.7)

# 저장 및 출력
plt.tight_layout()
plt.savefig(f'parking_results_spot_{target_spot_id}_icicr.png', dpi=300)
plt.show()
