import cv2
import numpy as np
import pandas as pd
import os

# --- [설정] ROI 영역 지정 ---
ROI_X1, ROI_X2 = 700, 850
ROI_Y1, ROI_Y2 = 550, 650

def get_std_dev(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    
    # img[Y1:Y2, X1:X2] 순서로 슬라이싱
    roi = img[ROI_Y1:ROI_Y2, ROI_X1:ROI_X2]
    
    # 표준편차 계산
    return np.std(roi)

def process_dataset(base_path):
    results = []
    
    for label in ['road', 'obstacle']:
        folder_path = os.path.join(base_path, label)
        if not os.path.exists(folder_path):
            print(f"경고: {folder_path} 폴더가 없습니다.")
            continue
            
        files = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg'))]
        print(f"{label} 데이터 분석 중... ({len(files)}개)")
        
        for file in files:
            std_val = get_std_dev(os.path.join(folder_path, file))
            if std_val is not None:
                # [수정] FileName 정보도 함께 저장!
                results.append({
                    'Type': label, 
                    'FileName': file, 
                    'StdDev': std_val
                })
                
    return pd.DataFrame(results)

if __name__ == "__main__":
    df = process_dataset('./dataset')
    
    # CSV 저장 (FileName 포함)
    df.to_csv('std_dev_values.csv', index=False)
    print("✅ 분석 완료! 'std_dev_values.csv'에 파일명이 포함되었습니다.")
    
    # 33 근처의 범인 이미지 찾기 (출력용)
    suspects = df[(df['Type'] == 'road') & (df['StdDev'] > 30)]
    if not suspects.empty:
        print("\n🚨 [범인 검거] Road 타입인데 30이 넘는 이미지들:")
        print(suspects)
