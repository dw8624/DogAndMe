# 🐕 나와 닮은 동물은? - AI 개 품종 예측 웹앱

> **FastAPI + PyTorch를 활용한 실시간 개 품종 분류 웹 애플리케이션**  
> 사용자의 얼굴 사진을 업로드하면 AI가 닮은 개 품종을 예측해주는 재미있는 웹서비스입니다.

![Demo](https://img.shields.io/badge/Demo-Live-success)
![Python](https://img.shields.io/badge/Python-3.8+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-Latest-green)
![PyTorch](https://img.shields.io/badge/PyTorch-2.7+-red)

## 🌟 프로젝트 특징

- **🤖 AI 기반 예측**: VGG16 모델로 133개 개 품종 분류
- **👤 얼굴 인식**: OpenCV를 활용한 실시간 얼굴 검출
- **📱 반응형 웹**: 모바일/데스크톱 완벽 호환
- **⚡ 빠른 응답**: FastAPI 비동기 처리로 빠른 예측

## 🚀 라이브 데모

**[🔗 웹사이트 바로가기](http://your-deployed-url.com)**

![screenshot](./assets/demo-screenshot.png)

## 🛠️ 기술 스택

### Backend
- **FastAPI**: 고성능 비동기 웹 프레임워크
- **PyTorch**: 딥러닝 모델 추론
- **OpenCV**: 컴퓨터 비전 및 얼굴 검출
- **Uvicorn**: ASGI 서버

### AI/ML
- **VGG16**: 사전 훈련된 CNN 모델
- **Transfer Learning**: 133개 개 품종 분류
- **Image Preprocessing**: PIL + torchvision transforms

### Frontend
- **Vanilla JavaScript**: 순수 자바스크립트
- **HTML5/CSS3**: 모던 웹 표준
- **Responsive Design**: 모바일 우선 디자인

## 📁 프로젝트 구조

```
FastAPIProject/
├── main.py                 # FastAPI 메인 애플리케이션
├── dog_classifier.pkl      # 훈련된 AI 모델 (539MB)
├── requirements.txt        # Python 의존성 패키지
├── haarcascades/          # 얼굴 검출용 cascade 파일들
├── assets/                # 이미지 및 문서 파일
├── venv/                  # Python 가상환경
└── README.md              # 프로젝트 문서
```

## ⚡ 빠른 시작

### 1. 저장소 클론
```bash
git clone https://github.com/dw8624/DogAndMe
cd DogAndMe
```

### 2. 가상환경 설정
```bash
python -m venv venv
# Windows
.\venv\Scripts\activate
# Mac/Linux  
source venv/bin/activate
```

### 3. 의존성 설치
```bash
pip install -r requirements.txt
```

### 4. 서버 실행
```bash
python main.py
# 또는
uvicorn main:app --reload
```

### 5. 브라우저에서 확인
- **웹사이트**: http://localhost:8000
- **API 문서**: http://localhost:8000/docs

## 🎯 주요 기능

### 1. 얼굴 검출 및 예측
- 업로드된 이미지에서 자동 얼굴 검출
- VGG16 기반 CNN으로 개 품종 예측
- 133개 품종 중 가장 유사한 품종 반환

### 2. 사용자 친화적 인터페이스
- 드래그 앤 드롭 파일 업로드
- 실시간 이미지 미리보기
- 로딩 상태 및 에러 처리

### 3. 모바일 최적화
- 반응형 디자인으로 모든 기기 지원
- 터치 친화적 인터페이스

## 🔧 API 엔드포인트

### `POST /predict`
이미지 파일을 업로드하여 개 품종 예측

**Request:**
```javascript
FormData: {
  file: <image_file>
}
```

**Response:**
```json
{
  "success": true,
  "breed": "Golden Retriever",
  "message": "You look like a Golden Retriever! 🐕"
}
```

### `GET /health`
서버 및 모델 상태 확인

**Response:**
```json
{
  "status": "healthy",
  "model_status": "loaded",
  "classes_loaded": 133,
  "pytorch_version": "2.7.1"
}
```


## 📊 성능 지표

- **모델 정확도**: ~85% (133개 품종 기준)
- **평균 예측 시간**: 2-3초
- **지원 이미지 형식**: JPG, PNG, WEBP
- **최대 파일 크기**: 10MB


### 새로운 품종 추가
모델 재훈련 후 `class_names` 리스트 업데이트

## 🤝 기여하기

1. 저장소 포크
2. 기능 브랜치 생성 (`git checkout -b feature/새기능`)
3. 변경사항 커밋 (`git commit -am '새기능 추가'`)
4. 브랜치에 푸시 (`git push origin feature/새기능`)
5. Pull Request 생성

## 📝 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 자세한 내용은 [LICENSE](LICENSE) 파일을 참고하세요.

## 🙏 참고 및 감사

### 원본 참고 코드
이 프로젝트는 다음 저장소의 CNN 모델 구조를 참고하여 개발되었습니다:
- **원본 저장소**: [Dog_Breed_Classifier_CNN](https://github.com/Tiwarim386/Dog_Breed_Classifier_CNN)
- **작성자**: Tiwarim386
- **참고 부분**: VGG16 기반 개 품종 분류 모델 구조

### 주요 개선사항
원본 코드를 기반으로 다음과 같은 개선을 진행했습니다:

1. **웹 애플리케이션 변환**
   - Jupyter Notebook → FastAPI 웹서비스
   - 실시간 웹 인터페이스 구현

2. **사용자 경험 개선**
   - 한국어 UI 및 핑크 테마 디자인
   - 드래그 앤 드롭 파일 업로드
   - 실시간 상태 표시 및 에러 처리

3. **성능 최적화**
   - 비동기 처리로 응답 속도 향상
   - 메모리 효율적인 모델 로딩
   - CPU/GPU 자동 선택

4. **배포 최적화**
   - Docker 컨테이너화
   - 클라우드 배포 지원
   - 상세한 문서화

### 사용된 오픈소스
- **PyTorch**: Facebook의 딥러닝 프레임워크
- **FastAPI**: Sebastián Ramirez의 웹 프레임워크
- **OpenCV**: Intel의 컴퓨터 비전 라이브러리

