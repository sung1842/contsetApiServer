from flask import Flask, render_template, request
import cv2
import numpy as np
import joblib
from deepface import DeepFace
import qrcode
import io
import base64

app = Flask(__name__)

# 모델 및 레이블 인코더 로드
model_path = 'models/animal_onlyface100_model.pkl'

label_encoder_path = 'models/label_onlyface100_encoder.pkl'

model = joblib.load(model_path)
le = joblib.load(label_encoder_path)

# 클래스 리스트
# 클래스 리스트 (한국어로)
classes = ['다람쥐', '고양이', '사슴', '공룡', '강아지', '여우', '말', '토끼', '거북이', '늑대']

# 얼굴 인식 함수
def detect_face(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    if len(faces) > 0:
        x, y, w, h = faces[0]
        face = img[y:y+h, x:x+w]
        face_path = "static/detected_face.jpg"
        cv2.imwrite(face_path, face)
        return face_path
    else:
        return None

# 이미지 분류 함수
def classify_image(image_path):
    face_path = detect_face(image_path)
    if face_path:
        embedding = DeepFace.represent(face_path, model_name="VGG-Face", enforce_detection=False)[0]["embedding"]
        embedding = np.array(embedding).reshape(1, -1)
        probabilities = model.predict_proba(embedding)[0]
        results = []
        for idx, prob in enumerate(probabilities):
            animal_class = classes[idx]
            animal_probability = prob * 100
            results.append({"animal": animal_class, "probability": animal_probability})
        return results
    else:
        return [{"error": "No face detected"}]

# URL 생성 함수
def generate_qr_code(url):
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=10,
        border=4,
    )
    qr.add_data(url)
    qr.make(fit=True)
    qr_image = qr.make_image(fill_color="black", back_color="white")
    # QR 코드 이미지를 base64로 인코딩하여 HTML에 직접 삽입 가능한 형태로 변환
    buffered = io.BytesIO()
    qr_image.save(buffered, format="PNG")
    qr_image_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return qr_image_str

@app.route('/', methods=['GET', 'POST'])
def index():
    # 현재 서버의 URL 생성
    server_url = request.host_url
    qr_image = generate_qr_code(server_url)

    if request.method == 'POST':
        # 파일 업로드 처리
        file = request.files['image']
        if file:
            image_path = "static/uploaded_image.jpg"
            file.save(image_path)
            prediction_results = classify_image(image_path)
            face_image = detect_face(image_path)
            return render_template('index.html', prediction_results=prediction_results, face_image=face_image, qr_image=qr_image)
    
    return render_template('index.html', prediction_results=None, face_image=None, qr_image=qr_image)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
