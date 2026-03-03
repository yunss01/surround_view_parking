import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
import seaborn as sns
import os

# --- [1. 라텍스 스타일 폰트 및 설정] ---
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10
})

# 1. 정답 좌표 (Spot 1 ~ 4)
IDEAL_POSES = {
    1: {'x': -3.203, 'y': 8.659},
    2: {'x': -3.203, 'y': 5.252},
    3: {'x': -3.203, 'y': 1.883},
    4: {'x': -3.203, 'y': -1.452}
}

spot_depth = 5.24  
spot_width = 3.40  

# 컬러 팔레트 통일
colors = {1: 'royalblue', 2: 'seagreen', 3: 'darkorange', 4: 'purple'}


# =====================================================================
# [Graph 1] 거리 오차 산점도 (1행 4열)
# =====================================================================
# obstacle 코드와 동일한 크기 적용 (18, 5)
fig1, axes1 = plt.subplots(1, 4, figsize=(18, 5)) 

for i, (spot_id, pose) in enumerate(IDEAL_POSES.items()):
    ax_z = axes1[i]
    tx, ty = pose['x'], pose['y']
    
    # 주차칸 선
    rect = Rectangle((ty - spot_width/2, tx - spot_depth/2), 
                     spot_width, spot_depth, linewidth=2, edgecolor='black', 
                     facecolor='none', linestyle='-')
    ax_z.add_patch(rect)
    
    # 15cm 성공 기준 원
    circle = Circle((ty, tx), 0.15, linewidth=2, edgecolor='red', facecolor='none', 
                    linestyle='dashed', label='Success Threshold (15cm)')
    ax_z.add_patch(circle)
    
    # 타겟 마커
    ax_z.plot(ty, tx, 'rX', markersize=12, linewidth=2, label='Target Center')
    
    # 데이터 로드 및 산점도
    file_name = f'parking_evaluation_spot_{spot_id}.csv'
    if os.path.exists(file_name):
        df = pd.read_csv(file_name)
        df = df[df['Total_Dist_Err(cm)'] < 100] # 아웃라이어 필터링
        ax_z.scatter(df['Final_Y'], df['Final_X'], c=colors[spot_id], alpha=0.6, s=40, edgecolors='k')
    
    zoom_margin = 0.4
    
    # X축 방향 반전! (큰 값 -> 작은 값)
    ax_z.set_xlim(ty + zoom_margin, ty - zoom_margin)
    ax_z.set_ylim(tx - zoom_margin, tx + zoom_margin)
    
    ax_z.set_title(f'Spot {spot_id}')
    ax_z.set_aspect('equal', adjustable='box')
    ax_z.grid(True, linestyle=':', alpha=0.7)
    
    # 개별 Y축 라벨 (첫 번째 그래프에만)
    if i == 0:
        ax_z.set_ylabel('X Coordinate [m]')
    else:
        ax_z.set_yticklabels([])

# 1. 범례 추출 및 중복 제거
handles, labels = axes1[0].get_legend_handles_labels()
by_label = dict(zip(labels, handles))

# 2. 범례 배치: obstacle 코드와 동일하게 상단 여백으로 올림 (1.05)
# scenarios_dist_err와 같은 모양을 위해 frameon=True를 유지합니다.
fig1.legend(by_label.values(), by_label.keys(), loc='upper center', 
            bbox_to_anchor=(0.5, 1.08), ncol=2, fontsize=12, frameon=True)

# 3. 공통 X축 라벨 
fig1.supxlabel('Y Coordinate [m]', fontsize=13, y=0.02)

# 4. 레이아웃 조정: 범례와 라벨을 모두 추가한 뒤 마지막에 기본값으로 호출
# (기존의 rect=[0, 0.03, 1, 0.90] 파라미터 제거)
plt.tight_layout()

# 5. 저장
plt.savefig('parking_dist_err_zoom_unified.pdf', format='pdf', bbox_inches='tight')
plt.savefig('parking_dist_err_zoom_unified.png', dpi=300, bbox_inches='tight')
plt.show()


# =====================================================================
# [Graph 2] 헤딩 오차 박스 플롯
# =====================================================================
data_list = []
for spot_id in [1, 2, 3, 4]:
    file_name = f'parking_evaluation_spot_{spot_id}.csv'
    if os.path.exists(file_name):
        df = pd.read_csv(file_name)
        df = df[df['Total_Dist_Err(cm)'] < 100] # 아웃라이어 필터링
        
        for h in df['Heading_Err(deg)'].values:
            data_list.append({'Spot': f'Spot {spot_id}', 'Heading Error (deg)': h})

df_heading = pd.DataFrame(data_list)

# obstacle 코드와 동일한 크기 적용 (10, 6)
fig2, ax2 = plt.subplots(figsize=(10, 6))

palette = [colors[1], colors[2], colors[3], colors[4]]

# 박스 플롯 그리기
sns.boxplot(x='Spot', y='Heading Error (deg)', data=df_heading, 
            palette=palette, ax=ax2, width=0.5, whis=(0, 100), showfliers=False)

# 5도 성공 기준선 (빨간 점선)
ax2.axhline(5.0, color='red', linestyle='dashed', linewidth=2, label='Success Threshold (5$^\circ$)')

ax2.set_ylabel('Heading Error (deg)')
ax2.set_xlabel('') # X축 'Spot 1~4' 라벨이 명확하므로 제목 생략
ax2.grid(True, linestyle=':', alpha=0.7, axis='y')

# [NEW] 범례를 오른쪽 상단에 배치
ax2.legend(loc='upper right')

# 불필요한 테두리 제거
sns.despine()
plt.tight_layout()

# PDF 및 PNG 저장
plt.savefig('parking_heading_err_boxplot.png', dpi=300, bbox_inches='tight')
plt.savefig('parking_heading_err_boxplot.pdf', format='pdf', bbox_inches='tight')
plt.show()
