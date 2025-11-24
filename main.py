import tensorflow as tf
from fastapi import FastAPI, UploadFile, File
from contextlib import asynccontextmanager
import numpy as np
import io
import librosa

# 1. 전역 변수로 두 모델 선언
image_model = None
audio_model = None

def preprocess_audio(audio_bytes):
    try:
        # 1. 바이트 데이터를 오디오 로드
        # sr=16000 (학습 데이터셋과 동일해야 함)
        audio_file = io.BytesIO(audio_bytes)
        y, sr = librosa.load(audio_file, sr=16000)
        
        # 2. MFCC 추출 (n_mfcc=40 학습 설정과 동일해야 함)
        # 기본 출력 shape: (n_mfcc, time) -> 예: (40, 92)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        
        # 3. 길이(Time) 고정: 모델이 100을 요구함
        target_length = 100
        current_length = mfcc.shape[1]

        if current_length < target_length:
            # 길이가 짧으면 뒤를 0으로 채움 (Padding)
            pad_width = target_length - current_length
            # ((0,0), (0, pad_width)) -> 앞 차원은 그대로, 뒤 차원(Time)만 패딩
            mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
        else:
            # 길이가 길면 100까지만 자름 (Truncating)
            mfcc = mfcc[:, :target_length]
            
        # 4. 차원 변환 (Transpose)
        # 현재 (40, 100) -> 모델 요구 (100, 40)
        mfcc = mfcc.T 
        
        # 5. 배치 차원 추가
        # (100, 40) -> (1, 100, 40)
        mfcc = np.expand_dims(mfcc, axis=0)

        return mfcc

    except Exception as e:
        print(f"전처리 오류: {e}")
        return None
# ==========================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 전역 변수를 가져옵니다
    global image_model, audio_model
    
    print("\n========== 서버 시작: 모델 로드 프로세스 ==========")

    # --- [1] 이미지 모델 로드 ---
    try:
        print("1. 이미지 모델(model_image.h5) 로드 중...")
        image_model = tf.keras.models.load_model("models/model_image.h5", compile=False)
        print("   ✅ 이미지 모델 로드 성공!")
    except Exception as e:
        print(f"   ❌ 이미지 모델 로드 실패: {e}")

    # --- [2] 오디오 모델 로드 ---
    try:
        print("2. 오디오 모델(model_audio.h5) 로드 중...")
        audio_model = tf.keras.models.load_model("models/model_audio.h5", compile=False)
        print("   ✅ 오디오 모델 로드 성공!")
    except Exception as e:
        print(f"   ❌ 오디오 모델 로드 실패: {e}")
    
    print("==================================================\n")
    
    yield  # 서버 실행 중...
    
    # --- [3] 종료 시 리소스 해제 ---
    image_model = None
    audio_model = None
    print("모든 모델 리소스 해제 완료.")

app = FastAPI(lifespan=lifespan)

@app.get("/")
def read_root():
    # 현재 로드 상태를 보여주는 메인 페이지
    return {
        "status": "Server is running",
        "models_status": {
            "image_model": "Loaded" if image_model else "Not Loaded",
            "audio_model": "Loaded" if audio_model else "Not Loaded"
        }
    }

# ---------------------------------------------------------
# 추론용 API
# ---------------------------------------------------------

@app.post("/predict/image")
async def predict_image(file: UploadFile = File(...)):
    if image_model is None:
        return {"error": "이미지 모델이 로드되지 않았습니다."}
    
    # 여기에 이미지 전처리 및 model.predict(image_model) 로직 작성
    return {"message": "이미지 모델 추론 기능 구현 필요"}

# --- [모듈 2] 2. /predict/audio 엔드포인트 구현 ---
@app.post("/predict/audio")
async def predict_audio(file: UploadFile = File(...)):
    """
    음성 파일을 받아 딥보이스 여부를 판별합니다.
    """
    # 1. 모델 로드 확인
    if audio_model is None:
        return {"error": "오디오 모델이 로드되지 않았습니다. 서버 로그를 확인하세요."}
    
    try:
        # 2. 파일 읽기 (바이트 단위)
        audio_bytes = await file.read()
        
        # 3. 전처리 (MFCC 추출)
        model_input = preprocess_audio(audio_bytes)
        
        if model_input is None:
            return {"error": "오디오 처리 중 오류가 발생했습니다. 파일 형식을 확인하세요."}
            
        # 4. 모델 추론 (predict)
        # 결과는 [[0.1234]] 형태의 2차원 배열로 나옴
        prediction = audio_model.predict(model_input)
        confidence = float(prediction[0][0]) # 0~1 사이의 확률 값
        
        # 5. 결과 판별 (임계값 0.5)
        # 0.5 이상이면 Fake(가짜), 미만이면 Real(진짜)
        is_deepfake = confidence > 0.5
        
        return {
            "filename": file.filename,
            "is_deepfake": is_deepfake,
            "confidence": f"{confidence:.4f}", # 소수점 4자리까지
            "message": "딥보이스 탐지 완료"
        }
        
    except Exception as e:
        return {"error": f"서버 내부 오류: {str(e)}"}