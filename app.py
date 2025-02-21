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

# from flask import Flask, request, jsonify
# import tensorflow as tf
# import numpy as np
# from flask_cors import CORS

# app = Flask(__name__)
# CORS(app)

# # Load the trained model
# MODEL_PATH = 'D:/regaphone - Copy (2)/Rega-Project/my_flask_app/uploads/yoga_pose_model2.h5'

# model = tf.keras.models.load_model(MODEL_PATH)

# # List of yoga pose names (based on your training data)
# POSE_NAMES = [
#     "assisted side bend", "Bird Dog Pose", "Bow pose", 
#     "Butterfly Pose_Bound Angle Pose (Baddha Konasana)", "Cactus Pose",
#     "Camel Pose", "Cat Cow Pose", "Chair Twist Pose", "Cow Face Pose",
#     "Downward Facing Dog Pose (Adho Mukha Svanasana)", "Eagle Pose (Garudasana)",
#     "Finger Up and Down Stretch", "Frog Pose", "Garland Pose",
#     "Gate yoga pose", "Goddess Pose", "Half Forward Bend",
#     "Half Moon Pose", "Half-Split Strength", "Head to Knee Forward Bend",
#     "Hero Pose", "Legs Up the Wall Pose (Viparita Karani)",
#     "Low Lunge Pose (Anjaneyasana)", "Lunging Calf Stretch",
#     "Mountain Pose (Tadasana)", "Paschimottanasana (The Seated Forward Bend)",
#     "Plow Pose", "Pyramid Pose", "Reverse warrior",
#     "Revolved Janu Sirsasana", "Revolved Side Angle Pose",
#     "Revolved Triangle Pose", "Sage Marichi Pose", "Salutation Seal",
#     "Side Neck Stretch", "Standing Bent Over Calf Strength",
#     "Standing Bow Pose", "Standing Figure Four Pose",
#     "Standing Forward Bend Pose (Uttanasana)", "Standing Quad Stretch Pose",
#     "Standing Spinal Twist Pose", "Sukhasana (Easy Pose)",
#     "Table to toe pose", "Tadasana with Lateral Bend",
#     "The Big Toe Pose", "Thread the Needle Pose",
#     "Triangle Pose (Trikonasana)", "Upward Salute",
#     "Virasana Hero Pose", "Vrikshasana (The Tree Pose)",
#     "Warrior I Pose (Virabhadrasana I)", "Warrior II Pose (Virabhadrasana II)",
#     "Wind Relieving Pose", "crescent lunge twist yoga pose",
#     "extended side angle pose", "half lord of the fishes pose",
#     "lizard pose"
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
        
#         # Get top 3 predictions
#         top_3_indices = np.argsort(prediction[0])[-3:][::-1]
#         top_3_predictions = [
#             {
#                 "pose": POSE_NAMES[idx],
#                 "confidence": float(prediction[0][idx])
#             } for idx in top_3_indices
#         ]
        
#         return jsonify({
#             "predicted_pose": POSE_NAMES[predicted_class],
#             "confidence": confidence,
#             "top_3_predictions": top_3_predictions
#         }), 200
            
#     except Exception as e:
#         print(f"Error in prediction: {str(e)}")
#         return jsonify({"error": str(e)}), 500

# if __name__ == '__main__':
#     print("Loading model and starting server...")
#     app.run(host='0.0.0.0', port=5000, debug=True)



from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# โหลดโมเดล LSTMyoga_pose_model_fold21225_5
MODEL_PATH = 'D:/regaphone - Copy (2)/Rega-Project/my_flask_app/uploads/yoga_pose_model_lstmt5.h5'
# MODEL_PATH = 'D:/regaphone - Copy (2)/Rega-Project/my_flask_app/uploads/yoga_pose_model_fold21225_5.h5'
model = tf.keras.models.load_model(MODEL_PATH)

# รายชื่อท่าโยคะทั้งหมดตามลำดับ
POSE_NAMES = [
    "Triangle Pose (Trikonasana)",
    "Warrior I Pose (Virabhadrasana I)",
    "Warrior II Pose (Virabhadrasana II)",
    "Standing Forward Bend Pose (Uttanasana)",
    "Vrikshasana (The Tree Pose)",
    "Downward Facing Dog Pose (Adho Mukha Svanasana)",
    "Utkatasana (The Chair Pose)",
    "Sukhasana (Easy Pose)",
    "Butterfly Pose_Bound Angle Pose (Baddha Konasana)",
    "Eagle Pose (Garudasana)",
    "Legs Up the Wall Pose (Viparita Karani)",
    "Low Lunge Pose (Anjaneyasana)",
    "Mountain Pose (Tadasana)",
    "Paschimottanasana (The Seated Forward Bend)",
    "Chair Twist Pose",
    "lizard pose",
    "crescent lunge twist yoga pose",
    "Revolved Janosha shansana",
    "wind relieving pose",
    "Bird Dog Pose",
    "Side Neck Stretch",
    "Half moon pose",
    "Gate yoga pose",
    "Upward Salute",
    "Table to toe pose",
    "Thread the needle pose",
    "Standing Quad Stretch Pose",
    "Tadasana with Lateral Bend",
    "Revolved Side Angle Pose",
    "Pyramid Pose",
    "Goddess Pose",
    "Reverse Warrior",
    "Standing Bow Pose",
    "Standing Figure Four Pose",
    "Revolved Triangle Pose",
    "Standing Spinal Twist Pose",
    "Lunging Calf Stretch",
    "Standing Bent Over Calf Strength",
    "Half-Split Strength",
    "frog pose",
    "Camel pose",
    "Cat cow pose",
    "Cow Face pose",
    "extended side angle pose",
    "Garland pose",
    "half lord of the fishes pose",
    "Hero pose",
    "Assisted side bend",
    "Finger Up and Down Strecth",
    "Salutation Seal",
    "Head to knee Forward Bend",
    "The big toe pose",
    "Sage Marichi",
    "Hald forward Bend",
    "Plow pose",
    "Bow pose",
    "Cactus pose"
]

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # รับ keypoints จาก request
        data = request.json
        keypoints = data.get('keypoints', [])
        
        if not keypoints:
            return jsonify({"error": "No keypoints provided"}), 400
            
        # แปลงเป็น numpy array และ reshape ให้เหมาะกับ LSTM
        keypoints = np.array(keypoints).reshape(1, 33, 3)
        
        # Predict
        prediction = model.predict(keypoints)
        predicted_class = np.argmax(prediction[0])
        confidence = float(np.max(prediction[0]))
        
        # หา top 3 predictions
        top_3_indices = np.argsort(prediction[0])[-3:][::-1]
        top_3_predictions = [
            {
                "pose": POSE_NAMES[idx],
                "confidence": float(prediction[0][idx])
            } for idx in top_3_indices
        ]
        
        return jsonify({
            "predicted_pose": POSE_NAMES[predicted_class],
            "confidence": confidence,
            "top_3_predictions": top_3_predictions
        }), 200
            
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print(f"Model loaded successfully. Ready to predict {len(POSE_NAMES)} yoga poses.")
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