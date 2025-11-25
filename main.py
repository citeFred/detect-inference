import tensorflow as tf
from fastapi import FastAPI, UploadFile, File
from contextlib import asynccontextmanager
import cv2
import numpy as np
import io
import librosa
from PIL import Image

# ==========================================
# [설정] 이미지 모델의 입력 크기 (가로, 세로)
TARGET_IMG_SIZE = (299, 299)  # Xception 기본 크기
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


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

# --- 이미지 전처리 함수 (얼굴 추출 기능 추가) ---
def preprocess_image(image_bytes):
    try:
        # 1. 바이트 -> Numpy 배열로 변환 (OpenCV 사용을 위해)
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # 2. 얼굴 탐지 (학습 때와 동일한 로직)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        # 3. 얼굴이 있으면 잘라냄(Crop), 없으면 전체 이미지 사용
        if len(faces) > 0:
            # 가장 큰 얼굴 하나만 가져오기 (보통 첫 번째나, 면적 계산 필요하지만 편의상 첫번째)
            (x, y, w, h) = faces[0]
            roi = image[y:y+h, x:x+w] # 얼굴 영역 Crop
            
            # RGB 변환 (OpenCV는 BGR임)
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            
            # PIL 이미지로 변환
            pil_image = Image.fromarray(roi)
            print("✅ 얼굴 검출 성공: 얼굴 영역을 모델에 입력합니다.")
        else:
            # 얼굴을 못 찾았으면 어쩔 수 없이 전체 이미지를 씀 (RGB 변환)
            print("⚠️ 얼굴 검출 실패: 전체 이미지를 사용합니다.")
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
        # 4. 크기 조정 (299x299)
        pil_image = pil_image.resize(TARGET_IMG_SIZE) # (299, 299)
        
        # 5. 정규화 및 차원 추가
        img_array = np.array(pil_image)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array

    except Exception as e:
        print(f"이미지 전처리 오류: {e}")
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

# --- [모듈 3] 1. /predict/image 엔드포인트 구현 ---
@app.post("/predict/image")
async def predict_image(file: UploadFile = File(...)):
    """
    이미지 파일을 받아 딥페이크 여부를 판별합니다.
    """
    if image_model is None:
        return {"error": "이미지 모델이 로드되지 않았습니다."}
    
    try:
        # 1. 파일 읽기
        image_bytes = await file.read()
        
        # 2. 전처리
        model_input = preprocess_image(image_bytes)
        
        if model_input is None:
            return {"error": "이미지 처리 중 오류가 발생했습니다. 올바른 이미지 파일인지 확인하세요."}
            
        # 3. 모델 추론
        prediction = image_model.predict(model_input)
        
        # 모델의 출력 형태에 따라 인덱싱이 다를 수 있음 (보통 이진 분류는 [0][0])
        confidence = float(prediction[0][0]) 
        
        # 4. 결과 판별 (0.5 기준)
        # 0.5 이상이면 Fake(딥페이크), 미만이면 Real(진짜)
        is_deepfake = confidence < 0.5
        
        return {
            "filename": file.filename,
            "is_deepfake": is_deepfake,
            "confidence": f"{confidence:.4f}",
            "message": "딥페이크 이미지 탐지 완료"
        }
        
    except Exception as e:
        # 팁: 여기서 shape mismatch 에러가 나면 TARGET_IMG_SIZE를 수정해야 함
        return {"error": f"서버 내부 오류: {str(e)}"}

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
        is_deepfake = confidence < 0.5
        
        return {
            "filename": file.filename,
            "is_deepfake": is_deepfake,
            "confidence": f"{confidence:.4f}", # 소수점 4자리까지
            "message": "딥보이스 탐지 완료"
        }
        
    except Exception as e:
        return {"error": f"서버 내부 오류: {str(e)}"}