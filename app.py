from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import joblib
from deepface import DeepFace
import qrcode
import io
import base64
import os

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
# 기존 코드의 detect_face 함수를 다음 코드로 대체합니다.

# 얼굴 인식 함수 (강화 버전: 패딩 추가 및 가장 큰 얼굴 선택)
def detect_face(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None # 이미지 로드 실패 시
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Haar Cascade 분류기 로드
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # 얼굴 감지 (파라미터는 기존 유지)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    if len(faces) > 0:
        # 1. 감지된 얼굴 중 가장 큰 얼굴을 선택하여 분석 (대표 얼굴이라고 가정)
        # DeepFace 분석은 하나의 얼굴만 처리할 때 가장 안정적입니다.
        (x, y, w, h) = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[0]
        
        # 2. 얼굴 주변에 패딩(여백)을 추가하여 자르기 영역 확장
        # 얼굴 너비(w)의 30%만큼 패딩을 추가
        padding = int(w * 0.3) 
        
        # 이미지 경계를 벗어나지 않도록 자르기 좌표 계산
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(img.shape[1], x + w + padding)
        y2 = min(img.shape[0], y + h + padding)
        
        # 3. 잘린 얼굴 이미지 추출
        face = img[y1:y2, x1:x2]
        
        # 4. 자른 이미지를 저장 (classify_image 함수에서 이 경로를 사용함)
        face_path = "static/detected_face.jpg" 
        cv2.imwrite(face_path, face)
        
        return face_path
    else:
        return None # 감지된 얼굴 없음

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

        try:
            # DeepFace.analyze를 사용하여 성별을 분석합니다.
            analysis = DeepFace.analyze(
                img_path=face_path, 
                actions=['gender'], 
                enforce_detection=False # detect_face에서 이미 자른 이미지이므로 검출 강제는 필요 없음
            )
            gender_result = analysis[0]['dominant_gender'] # 'Man' 또는 'Woman'
            gender_data = {"gender": gender_result}
        except Exception:
            # 분석 실패 시 기본값 설정
            gender_data = {"gender": "Unknown"}

        return {"predictions": results, "attributes": gender_data}
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

# 연예인 사진 목록을 가져오는 함수
def get_celebrity_matches(animal_type, gender):
    # 1. 대상 디렉토리 경로 구성 (예: 'static/celeb/고양이/Man')
    base_dir = os.path.join('static', 'celeb', animal_type, gender)

    if not os.path.exists(base_dir):
        print(f"경로를 찾을 수 없음: {base_dir}")
        return []

    celebrity_list = []

    try:
        photo_files = [f for f in os.listdir(base_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp', '.avif'))]

        for filename in photo_files[:2]:
            file_path = os.path.join(base_dir, filename)

            with open(file_path, 'rb') as f:
                image_bytes = f.read()
                # Base64 인코딩 후 디코딩하여 문자열로 만듭니다.
                base64_encoded_string = base64.b64encode(image_bytes).decode('utf-8')

                # 3. 연예인 이름 추출
                name_part = os.path.splitext(filename)[0]
                celeb_name = name_part.split('_')[0].split('.')[0]

                # 4. 이미지 MIME 타입 결정
                mime_type = f'image/{os.path.splitext(filename)[1].lower().lstrip(".")}'

                # 5. 프론트엔드에서 바로 사용할 수 있는 Data URL 형식 생성
                data_url = f'data:{mime_type};base64,{base64_encoded_string}'

                celebrity_list.append({
                    "name": celeb_name,
                    "photo_data": data_url # ⭐️ Base64 데이터 URL로 변경
                })

    except Exception as e:
        print(f"연예인 사진 Base64 인코딩 중 오류 발생: {e}")
        return []

    return celebrity_list

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

# API 전용 엔드포인트 추가 (Flask 파일에)
@app.route('/analyze', methods=['POST']) # Spring Boot의 .uri("/analyze")와 일치해야 함
def analyze_image():
    # 1. Spring Boot가 보낸 'image' 파일을 받습니다.
    if 'image' not in request.files:
        return jsonify({"error": "No image file part"}), 400
    
    file = request.files['image']

    UPLOAD_DIR = 'static'
    # static 디렉토리가 없으면 생성 (안전성 확보)
    if not os.path.exists(UPLOAD_DIR):
        os.makedirs(UPLOAD_DIR)
        
    temp_image_path = os.path.join(UPLOAD_DIR, "api_uploaded_image.jpg")
    file.save(temp_image_path) # 파일을 저장합니다.

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        # 3. 기존 AI 모델 함수 호출 및 결과 구조 변경 반영
        prediction_data = classify_image(temp_image_path)
        
        # 4. 얼굴 감지 실패 처리
        if "error" in prediction_data:
            # 얼굴 감지 실패 시
            return jsonify({"error": "AI Analysis Failed", "detail": prediction_data['error']}), 400
        
        # 5. 데이터 추출
        prediction_results = prediction_data['predictions']
        gender = prediction_data['attributes']['gender'] # ⭐️ 성별 데이터 추출
        
        # 가장 확률이 높은 동물상 추출 (예시)
        best_match = max(prediction_results, key=lambda x: x['probability'])
        animal_type = best_match['animal']

        # 연예인 매칭 함수 호출 및 결과 추출
        celeb_matches = get_celebrity_matches(animal_type, gender)
        
        # 6. 프론트엔드가 기대하는 최종 JSON 구조에 맞춰 반환
        final_response = {
            "animalType": animal_type,
            "animalDescription": f"당신은 {animal_type}상이며, 분석 확률은 {best_match['probability']:.2f}%입니다.",
            "gender": gender,
            "celebrities": celeb_matches,
            "recommendedStyle": "시크 앤 모던",
            "recommendedItems": [
                {"name": "추천 아이템 1", "image": "/static/celeb/강아지/Man/백현.webp", "link": "#"},
                {"name": "추천 아이템 2", "image": "/static/celeb/강아지/Man/백현.webp", "link": "#"}
            ]
        }
        
        return jsonify(final_response)
        
    except Exception as e:
        # AI 처리 중 예외 발생 시 Spring Boot에 500 에러를 반환
        print(f"AI Processing Error: {e}")
        return jsonify({"error": "AI Processing Failed", "detail": str(e)}), 500

    finally:
        # 6. 임시 저장 파일 삭제 (선택 사항: 서버 공간 확보)
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
