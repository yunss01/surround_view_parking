import os
from ultralytics import YOLO

def main():
    # 데이터셋 경로
    data_path = "/home/sukja/surround_view_parking_simulation/DL model/surround view parking.v9i.yolov8/data.yaml"

    print("🚀 파인튜닝(Hard Sample Mining)을 시작합니다!")
    
    # 모델 불러오기 (정상적으로 m 모델 불러옴)
    model = YOLO("yolov8m-pose.pt") 

    results = model.train(
        data=data_path,
        epochs=150,
        
        # --- 🚨 중요 수정 포인트 ---
        optimizer='AdamW',    # YOLO가 lr0를 무시하지 못하도록 옵티마이저를 강제로 지정!
        lr0=0.01,            # 낮춘 학습률이 이제 정상적으로 적용됩니다.
        batch=4,              # 8 -> 4 로 줄여서 CUDA out of memory(OOM) 에러 해결!
        # ---------------------------
        
        patience=50,
        imgsz=1280,
        device=0,
        
	# --- 🚨 극한의 Keypoint 타겟팅 옵션 ---
        pose=40.0,       # (기본 12.0) 점 위치 틀리는 것에 엄청난 페널티 부여!
        kobj=3.0,        # (기본 1.0) 점이 화면에 존재하는지(Objectness) 맞추는 데 가중치 3배!
        box=3.0,         # (기본 7.5) Bbox 크기 틀리는 건 대충 넘어가도록 대폭 하향!
        cls=1.0,         # (기본 0.5) 빈칸인지 아닌지 클래스 구분은 확실히 하도록 상향
    	
    	# --- Data Augmentation ---
        degrees=180.0,
        shear=15.0,
        perspective=0.0005,
        translate=0.2,
        scale=0.6,
        fliplr=0.0,
        
        project="parking_runs", 
        name="train_keypoint_focus",
        save=True               
    )

    print("\n🎉 파인튜닝 완료!")

if __name__ == '__main__':
    main()
