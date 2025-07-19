import pickle
import matplotlib.pyplot as plt
import cv2
import numpy as np
import torch
import torchvision.models as models
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms


def load_model_from_pkl(pkl_path):
    """pkl 파일에서 모델 로드"""
    with open(pkl_path, 'rb') as f:
        model_package = pickle.load(f)

    print("📦 Model package contents:")
    for key, value in model_package.items():
        if key != 'model_state_dict':  # state_dict는 너무 크니까 제외
            print(f"  {key}: {value}")

    return model_package


# 모델 패키지 로드
model_package = load_model_from_pkl('dog_classifier.pkl')


def rebuild_model_from_package(model_package, use_cuda=True):
    """패키지에서 모델 재구성"""

    # VGG16 모델 구조 재생성
    model = models.vgg16(pretrained=True)

    # 마지막 분류층을 133개 클래스로 변경
    n_inputs = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(n_inputs, model_package['num_classes'])

    # 저장된 가중치 로드
    model.load_state_dict(model_package['model_state_dict'])

    # 평가 모드로 설정
    model.eval()

    # GPU 사용 설정
    if use_cuda and torch.cuda.is_available():
        model = model.cuda()
        print("✅ Model loaded on GPU")
    else:
        print("✅ Model loaded on CPU")

    return model


# 모델 재구성
loaded_model = rebuild_model_from_package(model_package, use_cuda=torch.cuda.is_available())

# pkl에서 로드한 클래스 이름들을 전역 변수로 설정
class_names = model_package['class_names']
print(f"📝 Loaded {len(class_names)} class names")
print(f"First 5 classes: {class_names[:5]}")


def predict_breed_from_loaded_model(img_path, model, class_names, use_cuda=True):
    """로드된 모델로 품종 예측"""

    # 이미지 전처리 (VGG16용)
    def image_to_tensor(img_path):
        img = Image.open(img_path).convert('RGB')
        transformations = transforms.Compose([
            transforms.Resize(size=224),
            transforms.CenterCrop((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        return transformations(img)[:3, :, :].unsqueeze(0)

    # 이미지 전처리
    image_tensor = image_to_tensor(img_path)

    # GPU로 이동
    if use_cuda and torch.cuda.is_available():
        image_tensor = image_tensor.cuda()

    # 예측 수행
    with torch.no_grad():  # 메모리 효율성을 위해
        output = model(image_tensor)
        _, pred_tensor = torch.max(output, 1)
        pred = pred_tensor.cpu().numpy()[0] if use_cuda else pred_tensor.numpy()[0]

    return class_names[pred]

def load_convert_image_to_tensor(img_path):
    image = Image.open(img_path).convert('RGB')
    # resize to (244, 244) because VGG16 accept this shape
    in_transform = transforms.Compose([
                        transforms.Resize(size=(244, 244)),
                        transforms.ToTensor()]) # normalization .

    # discard the transparent, alpha channel (that's the :3) and add the batch dimension
    image = in_transform(image)[:3,:,:].unsqueeze(0)
    return image

def VGG16_predict(img_path):
    image_tensor = load_convert_image_to_tensor(img_path)
    VGG16 = models.vgg16(pretrained=True)
    use_cuda = torch.cuda.is_available()

    if use_cuda:
        VGG16 = VGG16.cuda()

    # move model inputs to cuda, if GPU available
    if use_cuda:
        image_tensor = image_tensor.cuda()

    # get sample outputs
    output = VGG16(image_tensor)
    # convert output probabilities to predicted class
    _, preds_tensor = torch.max(output, 1)
    pred = np.squeeze(preds_tensor.numpy()) if not use_cuda else np.squeeze(preds_tensor.cpu().numpy())

    return int(pred)  # predicted class index

def dog_detector(img_path):
    prediction = VGG16_predict(img_path)
    return ((prediction >= 151) & (prediction <= 268))  # true/false

def face_detector(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0


def run_app_with_loaded_model(img_path, model, class_names):
    """로드된 모델을 사용하는 run_app"""

    # 이미지 표시 함수
    def display_image(img_path, title="Title"):
        image = Image.open(img_path)
        plt.figure(figsize=(8, 6))
        plt.title(title, fontsize=14, fontweight='bold')
        plt.imshow(image)
        plt.axis('off')
        plt.show()

    print("🔍 Analyzing image...")

    # 강아지 검출
    if dog_detector(img_path):
        print("🐕 Hello Doggie!")
        predicted_breed = predict_breed_from_loaded_model(img_path, model, class_names)
        display_image(img_path, title=f"Predicted Breed: {predicted_breed}")
        print(f"🎯 Your breed is most likely: {predicted_breed.upper()}")

    # 사람 얼굴 검출
    elif face_detector(img_path):
        print("👨 Hello Human!")
        predicted_breed = predict_breed_from_loaded_model(img_path, model, class_names)
        display_image(img_path, title=f"You look like: {predicted_breed}")
        print(f"🎭 You look like a: {predicted_breed.upper()}")

    else:
        print("❌ Oh, we're sorry! We couldn't detect any dog or human face in the image.")
        display_image(img_path, title="No detection")
        print("🔄 Try another image!")

    print("=" * 50)

# 1. 모델 로드
print("📥 Loading model from pkl file...")
model_package = load_model_from_pkl('dog_classifier.pkl')

# 2. 모델 재구성
print("🔧 Rebuilding model...")
loaded_model = rebuild_model_from_package(model_package)

# 3. 클래스 이름 설정
class_names = model_package['class_names']

# 4. 테스트 실행
print("🎯 Testing loaded model...")

run_app_with_loaded_model("test11.jpg", loaded_model, class_names)