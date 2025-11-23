import tensorflow as tf
from fastapi import FastAPI, UploadFile, File
from contextlib import asynccontextmanager
import numpy as np

# 1. 전역 변수로 두 모델 선언
image_model = None
audio_model = None

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

@app.post("/predict/audio")
async def predict_audio(file: UploadFile = File(...)):
    if audio_model is None:
        return {"error": "오디오 모델이 로드되지 않았습니다."}
    
    # 여기에 오디오 전처리(Librosa 등) 및 model.predict(audio_model) 로직 작성
    return {"message": "오디오 모델 추론 기능 구현 필요"}