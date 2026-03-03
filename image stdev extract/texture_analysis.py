import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.lines import Line2D

# 1. 데이터 로드 (std_dev_values.csv가 있어야 함)
df = pd.read_csv('std_dev_values.csv')

# 2. 라텍스 스타일 폰트 설정
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "xtick.labelsize": 11,
})

plt.figure(figsize=(9, 5.5))

# 3. 히스토그램 그리기
ax = sns.histplot(data=df, x='StdDev', hue='Type', bins=30, 
                  palette={'road': 'royalblue', 'obstacle': 'darkorange'}, 
                  alpha=0.6, edgecolor='white', multiple='layer')

# 4. [수정됨] 임계값을 20.0으로 변경
plt.axvline(20.0, color='red', linestyle='dashed', linewidth=2.5)

# 5. X축 설정
plt.xticks(np.arange(0, 36, 2))
plt.xlim(0, 36)

# 6. [핵심] Y축 숫자 숨기기 & 라벨 변경
ax.set_yticks([])  # Y축의 눈금과 숫자를 완전히 지웁니다.
plt.ylabel('Relative Frequency')  # 'Sample Count'라는 말을 빼고 '상대 빈도'로 바꿈

# 7. 범례 박스 만들기 (Threshold 20.0으로 수정)
legend_elements = [
    Line2D([0], [0], color='royalblue', lw=4, label='Road Surface'),
    Line2D([0], [0], color='darkorange', lw=4, label='Obstacle'),
    Line2D([0], [0], color='red', lw=2, linestyle='--', label='Decision Threshold (20.0)')
]
plt.legend(handles=legend_elements, loc='upper right', frameon=True, shadow=True)

# 8. 타이틀 및 그리드
plt.title('Empirical Analysis of Texture Intensity for Obstacle Verification', pad=15)
plt.xlabel('Standard Deviation of Pixel Intensity')
# Y축 그리드는 숫자가 없으므로 없애거나 X축 그리드만 남기는 것이 깔끔합니다.
plt.grid(True, linestyle=':', alpha=0.5, axis='x') 

# 테두리 정리
sns.despine(left=True) # Y축 선 자체도 깔끔하게 지워줍니다.
plt.tight_layout()

# 9. 파일 저장
plt.savefig('texture_analysis_latex_v2.pdf', bbox_inches='tight')
plt.savefig('texture_analysis_latex_v2.png', dpi=300, bbox_inches='tight')

print("✅ 'texture_analysis_latex_v2.pdf' 저장이 완료되었습니다.")
plt.show()
