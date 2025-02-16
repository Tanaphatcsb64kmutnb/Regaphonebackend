# #/D:\regaphone - Copy (2)\Rega-Project\my_flask_app\app.py

# from flask import Flask, request, jsonify, render_template
# import tensorflow as tf
# import numpy as np
# import cv2
# import os
# import mediapipe as mp

# app = Flask(__name__)

# # Load the model
# model_path = 'D:/regaphone - Copy (2)/Rega-Project/my_flask_app/uploads/my_yoga_pose_model_v2.h5'
# if os.path.exists(model_path):
#     print(f"Loading model from {model_path}")
#     model = tf.keras.models.load_model(model_path)
# else:
#     raise FileNotFoundError(f"Model file not found at {model_path}")

# # Mediapipe pose detection setup
# mp_pose = mp.solutions.pose
# pose = mp_pose.Pose()

# # List of pose names as per the model's training order
# pose_names = ["downdog", "goddess", "plank", "tree", "warrior 2"]  # Adjust as needed

# @app.route('/')
# def index():
#     return render_template('index.html')  # Make sure you have an index.html file for the UI

# @app.route('/predict', methods=['POST'])
# def predict_pose():
#     try:
#         if 'image' not in request.files:
#             return jsonify({"error": "No image provided"}), 400

#         file = request.files['image']
#         file_bytes = np.frombuffer(file.read(), np.uint8)
#         image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

#         if image is None:
#             return jsonify({"error": "Invalid image"}), 400

#         # Resize the image for pose detection
#         image_resized = cv2.resize(image, (224, 224))
#         image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)

#         # Process the image using Mediapipe Pose
#         results = pose.process(image_rgb)

#         # Extract keypoints as input for the model
#         keypoints = []
#         if results.pose_landmarks:
#             for landmark in results.pose_landmarks.landmark:
#                 keypoints.append(landmark.x)
#                 keypoints.append(landmark.y)
#                 keypoints.append(landmark.z)

#         # Ensure the keypoints have the correct shape for the model
#         if len(keypoints) == 99:  # 33 keypoints * 3 dimensions (x, y, z)
#             keypoints = np.array(keypoints).reshape(1, -1)  # Shape: (1, 99)
#         else:
#             return jsonify({"error": "Invalid keypoints detected"}), 400

#         # Predict using the model
#         prediction = model.predict(keypoints)
#         predicted_class = np.argmax(prediction)

#         if predicted_class < len(pose_names):
#             predicted_pose_name = pose_names[predicted_class]
#             return jsonify({"predicted_pose": predicted_pose_name, "confidence": float(np.max(prediction))}), 200
#         else:
#             return jsonify({"error": "Prediction out of bounds"}), 500

#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000, debug=True)

















#predict version1
# from flask import Flask, request, jsonify
# import tensorflow as tf
# import numpy as np
# import cv2
# import os
# import mediapipe as mp
# from flask_cors import CORS

# app = Flask(__name__)
# CORS(app)

# # Load the model
# model_path = 'D:/regaphone - Copy (2)/Rega-Project/my_flask_app/uploads/my_yoga_pose_model_v2.h5'
# if os.path.exists(model_path):
#     print(f"Loading model from {model_path}")
#     model = tf.keras.models.load_model(model_path)
# else:
#     raise FileNotFoundError(f"Model file not found at {model_path}")

# # รายชื่อท่าโยคะทั้งหมด
# pose_names = [
#     "Bridge Pose (Setu Bandhasana)",
#     "Butterfly Pose_Bound Angle Pose (Baddha Konasana)", 
#     "Child's Pose (Balasana)",
#     "Cobra (Bhujangasana)",
#     "Downward Facing Dog Pose (Adho Mukha Svanasana)",
#     "Eagle Pose (Garudasana)",
#     "Half Lord of the Fishes Pose (Ardha Matsyendrasana)",
#     "Legs Up the Wall Pose (Viparita Karani)",
#     "Low Lunge Pose (Anjaneyasana)",
#     "Mountain Pose (Tadasana)",
#     "Paschimottanasana (The Seated Forward Bend)",
#     "Savasana",
#     "Sphinx Pose (Salamba Bhujangasana)",
#     "Standing Forward Bend Pose (Uttanasana)", 
#     "Sukhasana (Easy Pose)",
#     "Triangle Pose (Trikonasana)",
#     "Utkatasana (The Chair Pose)",
#     "Vrikshasana (The Tree Pose)",
#     "Warrior I Pose (Virabhadrasana I)",
#     "Warrior II Pose (Virabhadrasana II)"
# ]

# @app.route('/predict', methods=['POST'])
# def predict_pose():
#     try:
#         # Get keypoints data from request
#         data = request.json
#         keypoints = data.get('keypoints', [])
        
#         if not keypoints:
#             return jsonify({"error": "No keypoints provided"}), 400
            
#         # Convert to numpy array and reshape
#         keypoints = np.array(keypoints).reshape(1, -1)
        
#         # Ensure we have the correct number of keypoints (33 landmarks * 3 coordinates)
#         if keypoints.shape[1] != 99:
#             return jsonify({"error": f"Expected 99 values, got {keypoints.shape[1]}"}), 400
            
#         # Make prediction
#         prediction = model.predict(keypoints)
#         predicted_class = np.argmax(prediction)
#         confidence = float(np.max(prediction))
        
#         if predicted_class < len(pose_names):
#             return jsonify({
#                 "predicted_pose": pose_names[predicted_class],
#                 "confidence": confidence
#             }), 200
#         else:
#             return jsonify({"error": "Prediction out of bounds"}), 500
            
#     except Exception as e:
#         print(f"Error in prediction: {str(e)}")
#         return jsonify({"error": str(e)}), 500

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000, threaded=True, debug=True)



























# # predict model version 2
# from flask import Flask, request, jsonify
# import tensorflow as tf
# import numpy as np
# import cv2
# import os
# import mediapipe as mp
# from flask_cors import CORS

# app = Flask(__name__)
# CORS(app)

# # Load the model
# model_path = 'D:/regaphone - Copy (2)/Rega-Project/my_flask_app/uploads/yoga_pose_model2.h5'
# if os.path.exists(model_path):
#     print(f"Loading model from {model_path}")
#     model = tf.keras.models.load_model(model_path)
# else:
#     raise FileNotFoundError(f"Model file not found at {model_path}")

# # รายชื่อท่าโยคะทั้งหมด
# pose_names = [
#     "Cat Cow Pose",
#     "Half Lord of the Fishes Pose",
#     "Cactus Pose",
#     "Garland Pose",
#     "Frog Pose",
#     "Camel Pose",
#     "Virasana Hero Pose",
#     "Lunging Calf Stretch",
#     "Standing Bent Over Calf Strength",
#     "Half-Split Strength",
#     "Cow Face Pose",
#     "Assisted Side Bend",
#     "Finger Up and Down Stretch",
#     "Salutation Seal",
#     "Head to Knee Forward Bend",
#     "Extended Side Angle Pose",
#     "The Big Toe Pose",
#     "Sage Marichi Pose",
#     "Half Forward Bend",
#     "Plow Pose",
#     "Bow Pose",
#     "Thread the Needle Pose",
#     "Lizard Pose",
#     "Gate Pose",
#     "Revolved Janu Sirsasana",
#     "Bird Dog Pose",
#     "Side Neck Stretch",
#     "Spinal Twist",
#     "Upward Salute",
#     "Table to Toe Pose",
#     "Half Moon Pose",
#     "Crescent Lunge Twist",
#     "Wind Relieving Pose",
#     "Revolved Triangle Pose",
#     "Chair Twist Pose",
#     "Reverse Warrior",
#     "Goddess Pose",
#     "Pyramid Pose",
#     "Revolved Side Angle Pose",
#     "Tadasana with Lateral Bend",
#     "Standing Spinal Twist",
#     "Standing Quad Stretch",
#     "Standing Bow Pose",
#     "Standing Figure Four"
# ]

# @app.route('/predict', methods=['POST'])
# def predict_pose():
#     try:
#         # Get keypoints data from request
#         data = request.json
#         keypoints = data.get('keypoints', [])
        
#         if not keypoints:
#             return jsonify({"error": "No keypoints provided"}), 400
            
#         # Convert to numpy array and reshape
#         keypoints = np.array(keypoints).reshape(1, -1)
        
#         # Ensure we have the correct number of keypoints (33 landmarks * 3 coordinates)
#         if keypoints.shape[1] != 99:
#             return jsonify({"error": f"Expected 99 values, got {keypoints.shape[1]}"}), 400
            
#         # Make prediction
#         prediction = model.predict(keypoints)
#         predicted_class = np.argmax(prediction)
#         confidence = float(np.max(prediction))
        
#         if predicted_class < len(pose_names):
#             return jsonify({
#                 "predicted_pose": pose_names[predicted_class],
#                 "confidence": confidence,
#                 "class_index": int(predicted_class)
#             }), 200
#         else:
#             return jsonify({"error": "Prediction out of bounds"}), 500
            
#     except Exception as e:
#         print(f"Error in prediction: {str(e)}")
#         return jsonify({"error": str(e)}), 500

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000, threaded=True, debug=True)

# -*- coding: utf-8 -*-
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from collections import deque

app = Flask(__name__)
CORS(app, resources={
    r"/predict": {
        "origins": "*",
        "methods": ["POST"],
        "allow_headers": ["Content-Type"]
    }
})

# คอนฟิกพื้นฐาน
CONFIDENCE_THRESHOLD = 0.3
POSE_BUFFER_SIZE = 3
prediction_buffer = deque(maxlen=POSE_BUFFER_SIZE)

# ข้อมูลมุมมาตรฐานของแต่ละท่า
POSE_ANGLES = {
    "Tree Pose": {
        'right_shoulder': {'min': 80, 'max': 100},  # ไหล่ตรง
        'left_shoulder': {'min': 80, 'max': 100},   # ไหล่ตรง
        'right_elbow': {'min': 165, 'max': 195},    # แขนยกขึ้น
        'left_elbow': {'min': 165, 'max': 195},     # แขนยกขึ้น
        'right_hip': {'min': 165, 'max': 195},      # สะโพกตรง
        'left_hip': {'min': 165, 'max': 195},       # สะโพกตรง
        'right_knee': {'min': 165, 'max': 195},     # ขาที่ยืน
        'left_knee': {'min': 45, 'max': 90},        # ขาที่พับ
    }
}

# ข้อความ feedback สำหรับแต่ละท่า
POSE_FEEDBACK = {
    "Tree Pose": {
        'right_shoulder': {
            'too_low': "ยกไหล่ขวาขึ้นให้ตรง",
            'too_high': "ลดไหล่ขวาลงให้ตรง"
        },
        'left_shoulder': {
            'too_low': "ยกไหล่ซ้ายขึ้นให้ตรง",
            'too_high': "ลดไหล่ซ้ายลงให้ตรง"
        },
        'right_elbow': {
            'too_low': "ยกแขนขวาขึ้นอีกนิด",
            'too_high': "ลดแขนขวาลงเล็กน้อย"
        },
        'left_elbow': {
            'too_low': "ยกแขนซ้ายขึ้นอีกนิด",
            'too_high': "ลดแขนซ้ายลงเล็กน้อย"
        },
        'right_hip': {
            'too_low': "ยืดลำตัวให้ตรง",
            'too_high': "ผ่อนคลายลำตัวลงนิดหน่อย"
        },
        'left_hip': {
            'too_low': "ยืดลำตัวให้ตรง",
            'too_high': "ผ่อนคลายลำตัวลงนิดหน่อย"
        },
        'right_knee': {
            'too_low': "ยืดขาขวาให้ตรง",
            'too_high': "ย่อขาขวาลงเล็กน้อย"
        },
        'left_knee': {
            'too_low': "งอเข่าซ้ายมากขึ้น",
            'too_high': "ผ่อนเข่าซ้ายลงนิดหน่อย"
        }
    }
}

def normalize_keypoints(keypoints):
    """ปรับค่า keypoints ให้เป็นมาตรฐาน"""
    try:
        points = np.array(keypoints).reshape(-1, 3)
        center = np.mean(points, axis=0)
        centered_points = points - center
        
        # คำนวณระยะทางระหว่างจุดกลางสะโพกและไหล่
        hip_mid = (points[23] + points[24]) / 2
        shoulder_mid = (points[11] + points[12]) / 2
        spine_length = np.linalg.norm(shoulder_mid - hip_mid)
        
        normalized_points = centered_points / (spine_length + 1e-6)
        return normalized_points.flatten()
    except Exception as e:
        print(f"Error in normalize_keypoints: {str(e)}")
        return keypoints

def calculate_angles(landmarks):
    """คำนวณมุมของข้อต่อต่างๆ"""
    points = np.array(landmarks).reshape(-1, 3)
    
    def calculate_angle(p1, p2, p3):
        """คำนวณมุมระหว่าง 3 จุด"""
        a = np.array([p1[0], p1[1]])
        b = np.array([p2[0], p2[1]])
        c = np.array([p3[0], p3[1]])
        
        ba = a - b
        bc = c - b
        
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        
        return np.degrees(angle)
    
    # คำนวณมุมสำหรับแต่ละข้อต่อ
    angles = {
        'right_shoulder': calculate_angle(
            points[14],  # right elbow
            points[12],  # right shoulder
            points[24]   # right hip
        ),
        'left_shoulder': calculate_angle(
            points[13],  # left elbow
            points[11],  # left shoulder
            points[23]   # left hip
        ),
        'right_elbow': calculate_angle(
            points[12],  # right shoulder
            points[14],  # right elbow
            points[16]   # right wrist
        ),
        'left_elbow': calculate_angle(
            points[11],  # left shoulder
            points[13],  # left elbow
            points[15]   # left wrist
        ),
        'right_hip': calculate_angle(
            points[12],  # right shoulder
            points[24],  # right hip
            points[26]   # right knee
        ),
        'left_hip': calculate_angle(
            points[11],  # left shoulder
            points[23],  # left hip
            points[25]   # left knee
        ),
        'right_knee': calculate_angle(
            points[24],  # right hip
            points[26],  # right knee
            points[28]   # right ankle
        ),
        'left_knee': calculate_angle(
            points[23],  # left hip
            points[25],  # left knee
            points[27]   # left ankle
        )
    }
    
    return angles

def calculate_pose_score(pose_name, angles):
    """คำนวณคะแนนท่าทาง"""
    if pose_name not in POSE_ANGLES:
        return 0, []
        
    target_angles = POSE_ANGLES[pose_name]
    feedback = []
    score = 0
    total_points = len(target_angles)
    
    # เพิ่มความยืดหยุ่นในการตรวจจับ
    tolerance = 15  # องศา
    
    for joint, angle in angles.items():
        if joint in target_angles:
            target = target_angles[joint]
            min_angle = target['min'] - tolerance
            max_angle = target['max'] + tolerance
            
            if min_angle <= angle <= max_angle:
                score += 1
            else:
                if angle < min_angle:
                    feedback.append(POSE_FEEDBACK[pose_name][joint]['too_low'])
                else:
                    feedback.append(POSE_FEEDBACK[pose_name][joint]['too_high'])
    
    return (score / total_points) * 100, feedback

@app.route('/predict', methods=['POST'])
def predict_pose():
    try:
        data = request.json
        keypoints = data.get('keypoints', [])
        
        if not keypoints:
            response = jsonify({
                "error": "ไม่พบข้อมูล keypoints",
                "message": "กรุณาลองใหม่อีกครั้ง"
            })
            response.headers['Content-Type'] = 'application/json; charset=utf-8'
            return response, 400
        
        # ปรับค่า keypoints
        normalized_keypoints = normalize_keypoints(keypoints)
        
        # คำนวณมุมของข้อต่อต่างๆ
        angles = calculate_angles(normalized_keypoints)
        
        # หาท่าที่ตรงที่สุด
        best_pose = None
        max_score = 0
        best_feedback = []
        
        for pose_name in POSE_ANGLES.keys():
            score, feedback = calculate_pose_score(pose_name, angles)
            if score > max_score:
                max_score = score
                best_pose = pose_name
                best_feedback = feedback
        
        # สร้าง response
        if max_score >= CONFIDENCE_THRESHOLD * 100:
            result = {
                'predicted_pose': best_pose,
                'confidence': max_score / 100,
                'score': max_score,
                'feedback': best_feedback,
                'angles': angles
            }
        else:
            result = {
                'predicted_pose': 'ไม่สามารถระบุท่าได้',
                'confidence': 0,
                'score': 0,
                'feedback': ['กรุณาทำท่าให้ชัดเจนขึ้น'],
                'angles': angles
            }
        
        # เพิ่ม logging เพื่อดูค่ามุมต่างๆ
        print(f"Angles: {angles}")
        print(f"Score: {max_score}")
        print(f"Feedback: {best_feedback}")
        
        response = jsonify(result)
        response.headers['Content-Type'] = 'application/json; charset=utf-8'
        return response
            
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        response = jsonify({
            "error": f"เกิดข้อผิดพลาด: {str(e)}",
            "message": "กรุณาลองใหม่อีกครั้ง"
        })
        response.headers['Content-Type'] = 'application/json; charset=utf-8'
        return response, 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
# from flask import Flask, request, render_template, jsonify
# import tensorflow as tf
# import numpy as np
# import cv2
# import os
# import mediapipe as mp
# from werkzeug.utils import secure_filename

# app = Flask(__name__)

# # กำหนดโฟลเดอร์สำหรับเก็บรูปที่อัพโหลด
# UPLOAD_FOLDER = 'static/uploads'
# if not os.path.exists(UPLOAD_FOLDER):
#     os.makedirs(UPLOAD_FOLDER)
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# # โหลดโมเดล
# model_path = 'D:/regaphone - Copy (2)/Rega-Project/my_flask_app/uploads/yoga_pose_model2.h5'
# if os.path.exists(model_path):
#     print(f"กำลังโหลดโมเดลจาก {model_path}")
#     model = tf.keras.models.load_model(model_path)
# else:
#     raise FileNotFoundError(f"ไม่พบไฟล์โมเดลที่ {model_path}")

# # ตั้งค่า MediaPipe
# mp_pose = mp.solutions.pose
# pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)
# mp_drawing = mp.solutions.drawing_utils

# # รายชื่อท่าโยคะ (44 ท่า ตามโมเดลของคุณ)
# pose_names = [
#     "Cat Cow Pose",
#     "Half Lord of the Fishes Pose",
#     "Cactus Pose",
#     "Garland Pose",
#     "Frog Pose",
#     "Camel Pose",
#     "Virasana Hero Pose",
#     "Lunging Calf Stretch",
#     "Standing Bent Over Calf Strength",
#     "Half-Split Strength",
#     "Cow Face Pose",
#     "Assisted Side Bend",
#     "Finger Up and Down Stretch",
#     "Salutation Seal",
#     "Head to Knee Forward Bend",
#     "Extended Side Angle Pose",
#     "The Big Toe Pose",
#     "Sage Marichi Pose",
#     "Half Forward Bend",
#     "Plow Pose",
#     "Bow Pose",
#     "Thread the Needle Pose",
#     "Lizard Pose",
#     "Gate Pose",
#     "Revolved Janu Sirsasana",
#     "Bird Dog Pose",
#     "Side Neck Stretch",
#     "Spinal Twist",
#     "Upward Salute",
#     "Table to Toe Pose",
#     "Half Moon Pose",
#     "Crescent Lunge Twist",
#     "Wind Relieving Pose",
#     "Revolved Triangle Pose",
#     "Chair Twist Pose",
#     "Reverse Warrior",
#     "Goddess Pose",
#     "Pyramid Pose",
#     "Revolved Side Angle Pose",
#     "Tadasana with Lateral Bend",
#     "Standing Spinal Twist",
#     "Standing Quad Stretch",
#     "Standing Bow Pose",
#     "Standing Figure Four"
# ]

# def process_image(image_path):
#     try:
#         # อ่านรูปภาพ
#         image = cv2.imread(image_path)
#         if image is None:
#             return None, "ไม่สามารถอ่านรูปภาพได้"
        
#         # ปรับขนาดรูปภาพถ้าจำเป็น
#         height, width = image.shape[:2]
#         if width > 1920:  # ถ้ารูปใหญ่เกินไป ให้ปรับขนาดลง
#             scale = 1920 / width
#             width = 1920
#             height = int(height * scale)
#             image = cv2.resize(image, (width, height))
        
#         # แปลงเป็น RGB สำหรับ MediaPipe
#         image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
#         # ตรวจจับท่าทาง
#         results = pose.process(image_rgb)
        
#         if not results.pose_landmarks:
#             return None, "ไม่พบตำแหน่งจุดสำคัญบนร่างกาย กรุณาถ่ายรูปให้เห็นร่างกายชัดเจน"
        
#         # เก็บ landmarks
#         keypoints = []
#         for landmark in results.pose_landmarks.landmark:
#             keypoints.extend([landmark.x, landmark.y, landmark.z])
        
#         # แปลงเป็น numpy array
#         keypoints = np.array(keypoints).reshape(1, -1)
        
#         # ตรวจสอบขนาดของ input
#         if keypoints.shape[1] != 99:  # 33 landmarks * 3 coordinates
#             return None, f"จำนวนจุดไม่ถูกต้อง (พบ {keypoints.shape[1]} ค่า, ต้องการ 99 ค่า)"
        
#         # ทำนายท่าโยคะ
#         prediction = model.predict(keypoints)
#         predicted_class = np.argmax(prediction[0])
        
#         # ตรวจสอบว่า index ไม่เกินขนาดของ list
#         if predicted_class >= len(pose_names):
#             return None, f"ผลการทำนายไม่ถูกต้อง (class {predicted_class} เกินจำนวนท่าที่มี)"
        
#         # หา top 3 predictions
#         top_3_indices = np.argsort(prediction[0])[-3:][::-1]
#         top_3_predictions = [
#             {
#                 "pose": pose_names[idx],
#                 "confidence": float(prediction[0][idx])
#             } for idx in top_3_indices
#         ]
        
#         # สร้าง output
#         result = {
#             "predicted_pose": pose_names[predicted_class],
#             "confidence": float(prediction[0][predicted_class]),
#             "top_3": top_3_predictions
#         }
        
#         return result, None
        
#     except Exception as e:
#         return None, f"เกิดข้อผิดพลาด: {str(e)}"

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     if 'image' not in request.files:
#         return render_template('index.html', error="กรุณาเลือกรูปภาพ")
    
#     file = request.files['image']
#     if file.filename == '':
#         return render_template('index.html', error="ไม่ได้เลือกไฟล์")
    
#     if file:
#         filename = secure_filename(file.filename)
#         filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#         file.save(filepath)
        
#         result, error = process_image(filepath)
        
#         if error:
#             return render_template('index.html', error=error)
        
#         return render_template('index.html',
#                              result=result,
#                              image_path=f'uploads/{filename}')

# if __name__ == '__main__':
#     app.run(debug=True)