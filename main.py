from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import pickle
import cv2
import numpy as np
import torch
import torchvision.models as models
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import io
import os
from typing import Optional
import base64

app = FastAPI(title="Dog Breed Predictor", description="Find out which dog breed you look like!")

# 전역 변수들
loaded_model = None
class_names = None


def load_model_from_pkl(pkl_path: str):
    """pkl 파일에서 모델 로드 - Custom unpickler 사용"""
    print(f"🔍 Loading model from {pkl_path}")
    print(f"🔍 PyTorch version: {torch.__version__}")
    print(f"🔍 CUDA available: {torch.cuda.is_available()}")

    try:
        print("📥 Loading with custom CPU unpickler...")
        import pickle
        import io

        class CPUUnpickler(pickle.Unpickler):
            def find_class(self, module, name):
                if module == 'torch.storage' and name == '_load_from_bytes':
                    return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
                else:
                    return super().find_class(module, name)

        with open(pkl_path, 'rb') as f:
            model_package = CPUUnpickler(f).load()

        print(f"✅ Model loaded successfully!")
        return model_package

    except FileNotFoundError:
        print(f"❌ Model file {pkl_path} not found")
        return None
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None


def rebuild_model_from_package(model_package: dict, use_cuda: bool = True):
    """패키지에서 모델 재구성"""
    if model_package is None:
        return None

    try:
        # VGG16 모델 구조 재생성
        model = models.vgg16(pretrained=True)

        # 마지막 분류층을 133개 클래스로 변경
        n_inputs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(n_inputs, model_package['num_classes'])

        # 저장된 가중치 로드 (CPU 환경 고려)
        state_dict = model_package['model_state_dict']

        # GPU에서 저장된 모델을 CPU에서 로드하는 경우 처리
        if not torch.cuda.is_available() and use_cuda:
            # GPU 텐서를 CPU로 변환
            cpu_state_dict = {}
            for key, value in state_dict.items():
                if torch.is_tensor(value):
                    cpu_state_dict[key] = value.cpu()
                else:
                    cpu_state_dict[key] = value
            state_dict = cpu_state_dict
            use_cuda = False  # CUDA 사용 불가하므로 False로 설정

        model.load_state_dict(state_dict)

        # 평가 모드로 설정
        model.eval()

        # GPU 사용 설정
        if use_cuda and torch.cuda.is_available():
            model = model.cuda()
            print("✅ Model loaded on GPU")
        else:
            print("✅ Model loaded on CPU")

        return model
    except Exception as e:
        print(f"❌ Error rebuilding model: {e}")
        return None


def face_detector(image_bytes: bytes) -> bool:
    """이미지에서 얼굴 검출"""
    try:
        # bytes를 numpy array로 변환
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return False

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # OpenCV의 기본 얼굴 검출기 사용 (haarcascade 파일이 없을 경우 대비)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        return len(faces) > 0
    except Exception as e:
        print(f"❌ Face detection error: {e}")
        return False


def predict_breed_from_image(image_bytes: bytes) -> Optional[str]:
    """이미지에서 개 품종 예측"""
    if loaded_model is None or class_names is None:
        return None

    try:
        # bytes를 PIL Image로 변환
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

        # 이미지 전처리 (VGG16용)
        transformations = transforms.Compose([
            transforms.Resize(size=224),
            transforms.CenterCrop((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        image_tensor = transformations(image)[:3, :, :].unsqueeze(0)

        # GPU로 이동 (사용 가능한 경우에만)
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            try:
                image_tensor = image_tensor.cuda()
            except Exception as e:
                print(f"⚠️ Failed to move tensor to GPU: {e}. Using CPU instead.")
                use_cuda = False

        # 예측 수행
        with torch.no_grad():
            output = loaded_model(image_tensor)
            _, pred_tensor = torch.max(output, 1)
            pred = pred_tensor.cpu().numpy()[0] if use_cuda else pred_tensor.numpy()[0]

        return class_names[pred]
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return None


# 앱 시작시 모델 로드
@app.on_event("startup")
async def startup_event():
    global loaded_model, class_names

    print("🚀 Starting up FastAPI Dog Breed Predictor...")

    # 모델 로드
    model_package = load_model_from_pkl('dog_classifier.pkl')
    if model_package:
        # CPU 환경 고려하여 모델 재구성
        use_cuda = torch.cuda.is_available()
        loaded_model = rebuild_model_from_package(model_package, use_cuda=use_cuda)
        class_names = model_package['class_names']
        print(f"📝 Loaded {len(class_names)} class names")

        if not use_cuda:
            print("⚠️ CUDA not available. Running on CPU mode.")
    else:
        print("⚠️ Failed to load model. Some features may not work.")


@app.get("/", response_class=HTMLResponse)
async def main():
    """메인 페이지"""
    html_content = """
    <!DOCTYPE html>
    <html lang="ko">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>나와 닮은 동물은?</title>
        <style>
            body {
                font-family: 'Arial', sans-serif;
                background-color: #FFF4F2;
                text-align: center;
                padding: 20px;
            }
            h1 {
                color: #FF4C8B;
                font-size: 2.5em;
            }
            .subheading {
                background-color: #FFB5A7;
                color: white;
                padding: 10px;
                margin-bottom: 30px;
                font-size: 1.2em;
                border-radius: 8px;
            }
            .upload-form {
                background-color: #FFE5DC;
                border: 3px dashed #FFB5A7;
                padding: 30px;
                border-radius: 15px;
                display: inline-block;
                cursor: pointer;
                transition: all 0.3s ease;
            }
            .upload-form:hover {
                border-color: #FF6F91;
                background-color: #FFD5CC;
            }
            input[type="file"] {
                display: none;
            }
            .upload-btn {
                background-color: #FF6F91;
                color: white;
                border: none;
                padding: 15px 30px;
                font-size: 1.1em;
                border-radius: 10px;
                cursor: pointer;
                margin-top: 20px;
                transition: background-color 0.3s;
            }
            .upload-btn:hover {
                background-color: #FF4C8B;
            }
            .upload-btn:disabled {
                background-color: #ccc;
                cursor: not-allowed;
            }
            .preview-img {
                max-width: 300px;
                max-height: 300px;
                border-radius: 15px;
                margin: 20px auto;
                display: block;
                box-shadow: 0 4px 8px rgba(0,0,0,0.2);
            }
            .result {
                margin-top: 30px;
                padding: 20px;
                background-color: white;
                border-radius: 15px;
                box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                display: inline-block;
            }
            .result h2 {
                color: #FF4C8B;
                font-size: 2em;
                margin-bottom: 20px;
            }
            .result-animal {
                font-size: 2.5em;
                color: #FF6F91;
                font-weight: bold;
                margin: 20px 0;
            }
            .loading {
                color: #FF6F91;
                font-size: 1.2em;
                margin: 20px 0;
            }
            .error {
                color: #ff4444;
                background-color: #ffe6e6;
                border: 2px solid #ff4444;
                padding: 15px;
                border-radius: 10px;
                margin: 20px auto;
                display: inline-block;
            }
            .survey {
                margin-top: 50px;
                background-color: #fff;
                border-radius: 15px;
                padding: 20px;
                display: inline-block;
                box-shadow: 0 0 10px rgba(0,0,0,0.1);
            }
            .survey a {
                display: inline-block;
                margin-top: 10px;
                background-color: #FF4C8B;
                color: white;
                padding: 10px 20px;
                border-radius: 8px;
                text-decoration: none;
                font-weight: bold;
            }
            .survey a:hover {
                background-color: #FF6F91;
            }
            .status-indicator {
                position: fixed;
                top: 20px;
                right: 20px;
                padding: 10px 15px;
                border-radius: 10px;
                font-size: 0.9em;
                font-weight: bold;
            }
            .status-ok {
                background: rgba(76, 175, 80, 0.9);
                color: white;
            }
            .status-error {
                background: rgba(244, 67, 54, 0.9);
                color: white;
            }
            @media (max-width: 600px) {
                .preview-img {
                    width: 90%;
                }
                .upload-form {
                    width: 90%;
                }
            }
        </style>
    </head>
    <body>
        <div id="statusIndicator" class="status-indicator">모델 상태 확인 중...</div>
        
        <h1>나와 닮은 동물은? 🐕</h1>
        <div class="subheading">나는 어떤 동물과 닮았을까? 사진을 업로드 하면, 나와 닮은 동물을 확인할 수 있습니다.</div>
        
        <div class="upload-form" onclick="document.getElementById('fileInput').click()">
            <p>📷 여기를 클릭해서 사진을 업로드하세요</p>
            <p style="font-size: 0.9em; color: #666;">JPG, PNG 파일 지원</p>
            <input type="file" id="fileInput" accept="image/*" onchange="previewImage()">
        </div>

        <div id="preview"></div>

        <button class="upload-btn" id="analyzeBtn" onclick="analyzeImage()" disabled>
            🔍 동물 찾기!
        </button>

        <div id="result"></div>

        <div class="survey">
            <p>설문조사에 참여해주시면<br>추첨을 통해 소정의 기프티콘을 드립니다.<br>감사합니다 😊</p>
            <a href="https://docs.google.com/forms/d/e/1FAIpQLSfM0CURwgynFKiXDLbLxwsHoBIdyhgKRsPZrGSlI-_ScEU1NA/viewform" target="_blank">설문조사 바로가기</a>
        </div>

        <script>
            let selectedFile = null;

            // 페이지 로드 시 모델 상태 확인
            window.onload = function() {
                checkModelStatus();
            };

            async function checkModelStatus() {
                const statusIndicator = document.getElementById('statusIndicator');
                try {
                    const response = await fetch('/health');
                    const status = await response.json();
                    
                    if (status.model_status === 'loaded') {
                        statusIndicator.textContent = '✅ AI 준비완료';
                        statusIndicator.className = 'status-indicator status-ok';
                    } else {
                        statusIndicator.textContent = '❌ AI 로딩 중';
                        statusIndicator.className = 'status-indicator status-error';
                    }
                } catch (error) {
                    statusIndicator.textContent = '❌ 서버 오류';
                    statusIndicator.className = 'status-indicator status-error';
                }
            }

            function previewImage() {
                const fileInput = document.getElementById('fileInput');
                const preview = document.getElementById('preview');
                const analyzeBtn = document.getElementById('analyzeBtn');

                if (fileInput.files && fileInput.files[0]) {
                    selectedFile = fileInput.files[0];
                    const reader = new FileReader();

                    reader.onload = function(e) {
                        preview.innerHTML = '<img src="' + e.target.result + '" class="preview-img" alt="미리보기">';
                        analyzeBtn.disabled = false;
                    }

                    reader.readAsDataURL(fileInput.files[0]);
                }
            }

            async function analyzeImage() {
                if (!selectedFile) {
                    alert('먼저 사진을 선택해주세요!');
                    return;
                }

                const resultDiv = document.getElementById('result');
                const analyzeBtn = document.getElementById('analyzeBtn');

                // 로딩 상태
                resultDiv.innerHTML = '<div class="loading">🔍 사진을 분석 중입니다... 잠시만 기다려주세요!</div>';
                analyzeBtn.disabled = true;

                const formData = new FormData();
                formData.append('file', selectedFile);

                try {
                    const response = await fetch('/predict', {
                        method: 'POST',
                        body: formData
                    });

                    const result = await response.json();

                    if (response.ok) {
                        if (result.success) {
                            resultDiv.innerHTML = `
                                <div class="result">
                                    <h2>🎉 분석 완료!</h2>
                                    <p>당신과 닮은 동물은:</p>
                                    <div class="result-animal">${result.breed}</div>
                                    <p style="color: #666; margin-top: 20px;">
                                        이 품종이 당신의 특징과 가장 잘 어울려요! 🐾
                                    </p>
                                </div>
                            `;
                        } else {
                            resultDiv.innerHTML = `
                                <div class="error">
                                    <h3>😅 ${result.message}</h3>
                                    <p>얼굴이 선명하게 나온 사진을 다시 업로드해주세요!</p>
                                </div>
                            `;
                        }
                    } else {
                        throw new Error(result.detail || '알 수 없는 오류가 발생했습니다.');
                    }
                } catch (error) {
                    resultDiv.innerHTML = `
                        <div class="error">
                            <h3>❌ 오류가 발생했습니다</h3>
                            <p>${error.message}</p>
                            <p>잠시 후 다시 시도해주세요.</p>
                        </div>
                    `;
                } finally {
                    analyzeBtn.disabled = false;
                }
            }
        </script>
    </body>
    </html>
    
    """
    return HTMLResponse(content=html_content)


@app.post("/predict")
async def predict_dog_breed(file: UploadFile = File(...)):
    """이미지 업로드 후 개 품종 예측"""

    # 파일 형식 확인
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="Please upload an image file")

    try:
        # 이미지 읽기
        image_bytes = await file.read()

        # 모델이 로드되었는지 확인
        if loaded_model is None:
            raise HTTPException(status_code=500, detail="Model not loaded. Please check server logs.")

        # 얼굴 검출
        if not face_detector(image_bytes):
            return JSONResponse(content={
                "success": False,
                "message": "No human face detected in the image! 👤❌"
            })

        # 개 품종 예측
        predicted_breed = predict_breed_from_image(image_bytes)

        if predicted_breed is None:
            raise HTTPException(status_code=500, detail="Failed to predict breed")

        return JSONResponse(content={
            "success": True,
            "breed": predicted_breed,
            "message": f"You look like a {predicted_breed}! 🐕"
        })

    except Exception as e:
        print(f"❌ Error in prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/health")
async def health_check():
    """서버 상태 확인"""
    model_status = "loaded" if loaded_model is not None else "not loaded"
    return {
        "status": "healthy",
        "model_status": model_status,
        "classes_loaded": len(class_names) if class_names else 0
    }


if __name__ == "__main__":
    import uvicorn

    print("🚀 Starting Dog Breed Predictor Server...")
    print("📍 Server will be available at: http://localhost:8000")
    print("📋 API docs will be available at: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)