import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import seaborn as sns
import os

# --- [1. 스타일 설정] ---
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10
})

# 1. 시나리오 정의 (순서대로)
scenarios = [
    ("No Obstacle", "parking_evaluation_spot_2.csv", "royalblue"),
    ("Obstacle 1", "parking_evaluation_obstacle1_spot_2.csv", "seagreen"),
    ("Obstacle 3", "parking_evaluation_obstacle3_spot_2.csv", "darkorange"),
    ("Obstacle 1 & 3", "parking_evaluation_obstacle1&3_spot_2.csv", "purple")
]

# 2. 정답 좌표 (Spot 2)
target_x = -3.203  # Depth (세로축)
target_y = 5.252   # Lateral (가로축)

# --- [Graph 1] 거리 오차 산점도 (1행 4열) ---
fig1, axes1 = plt.subplots(1, 4, figsize=(18, 5)) # 가로로 넉넉하게

for i, (name, filename, color) in enumerate(scenarios):
    ax = axes1[i]
    
    # 15cm 성공 기준 원 (중심: target_y, target_x)
    circle = Circle((target_y, target_x), 0.15, linewidth=2, edgecolor='red', facecolor='none', linestyle='dashed', label='Success Threshold (15cm)')
    ax.add_patch(circle)
    
    # Target Marker
    ax.plot(target_y, target_x, 'rX', markersize=12, linewidth=2, label='Target Center')
    
    if os.path.exists(filename):
        df = pd.read_csv(filename)
        # 최신 100회 데이터만 사용
        if len(df) > 100: df = df.tail(100)
        
        # 산점도 그리기
        ax.scatter(df['Final_Y'], df['Final_X'], c=color, alpha=0.6, s=40, edgecolors='k')
    
    zoom = 0.4
    
    # X축 방향 반전! (큰 값 -> 작은 값)
    ax.set_xlim(target_y + zoom, target_y - zoom) 
    ax.set_ylim(target_x - zoom, target_x + zoom)
    
    ax.set_title(name)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, linestyle=':', alpha=0.7)
    
    # [수정] 개별 X축 라벨 삭제!
    if i == 0:
        ax.set_ylabel('X Coordinate [m]')

# [NEW] 4개의 그래프를 모두 아우르는 그림 전체의 통합 범례 (상단 중앙 배치)
handles, labels = axes1[0].get_legend_handles_labels()
# 중복 없는 범례 추출
by_label = dict(zip(labels, handles))
fig1.legend(by_label.values(), by_label.keys(), loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=2, fontsize=12)

# [NEW] 전체 서브플롯 하단 중앙에 공통 X축 라벨 추가
fig1.supxlabel('Y Coordinate [m]', fontsize=13, y=0.02)

plt.tight_layout()

# PNG와 PDF 저장
plt.savefig('scenarios_dist_err.png', dpi=300, bbox_inches='tight')
plt.savefig('scenarios_dist_err.pdf', format='pdf', bbox_inches='tight')
plt.show()

# --- [Graph 2] 헤딩 오차 박스 플롯 ---
data_list = []
for name, filename, color in scenarios:
    if os.path.exists(filename):
        df = pd.read_csv(filename)
        if len(df) > 100: df = df.tail(100)
        
        for h in df['Heading_Err(deg)']:
            data_list.append({'Scenario': name, 'Heading Error (deg)': h})

df_heading = pd.DataFrame(data_list)

fig2, ax2 = plt.subplots(figsize=(10, 6))
palette = [s[2] for s in scenarios] # 위에서 정의한 색상 그대로 사용

# 박스플롯 그리기
sns.boxplot(x='Scenario', y='Heading Error (deg)', data=df_heading, 
            palette=palette, ax=ax2, width=0.5, whis=(0, 100), showfliers=False)

# 5도 성공 기준선
ax2.axhline(5.0, color='red', linestyle='dashed', linewidth=2, label='Success Threshold (5 deg)')

ax2.set_xlabel('')
ax2.set_ylabel('Heading Error (deg)')
ax2.grid(True, linestyle=':', alpha=0.7)
ax2.legend(loc='upper right')

plt.tight_layout()

# PNG와 PDF 저장
plt.savefig('scenarios_heading_err.png', dpi=300, bbox_inches='tight')
plt.savefig('scenarios_heading_err.pdf', format='pdf', bbox_inches='tight')
plt.show()
