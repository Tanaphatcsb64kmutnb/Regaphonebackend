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





# from flask import Flask, request, jsonify
# import tensorflow as tf
# import numpy as np
# from flask_cors import CORS

# app = Flask(__name__)
# CORS(app)

# # # โหลดโมเดล LSTMyoga_pose_model_fold21225_5
# # MODEL_PATH = 'D:/regaphone - Copy (2)/Rega-Project/my_flask_app/uploads/yoga_pose_model_lstmt5.h5'
# MODEL_PATH = 'D:/regaphone - Copy (2)/Rega-Project/my_flask_app/uploads/yoga_pose_model_best_folddd28268.h5'

# model = tf.keras.models.load_model(MODEL_PATH)

# # # รายชื่อท่าโยคะทั้งหมดตามลำดับ
# # POSE_NAMES = [
# #     "Triangle Pose (Trikonasana)",
# #     "Warrior I Pose (Virabhadrasana I)",
# #     "Warrior II Pose (Virabhadrasana II)",
# #     "Standing Forward Bend Pose (Uttanasana)",
# #     "Vrikshasana (The Tree Pose)",
# #     "Downward Facing Dog Pose (Adho Mukha Svanasana)",
# #     "Utkatasana (The Chair Pose)",
# #     "Sukhasana (Easy Pose)",
# #     "Butterfly Pose_Bound Angle Pose (Baddha Konasana)",
# #     "Eagle Pose (Garudasana)",
# #     "Legs Up the Wall Pose (Viparita Karani)",
# #     "Low Lunge Pose (Anjaneyasana)",
# #     "Mountain Pose (Tadasana)",
# #     "Paschimottanasana (The Seated Forward Bend)",
# #     "Chair Twist Pose",
# #     "lizard pose",
# #     "crescent lunge twist yoga pose",
# #     "Revolved Janosha shansana",
# #     "wind relieving pose",
# #     "Bird Dog Pose",
# #     "Side Neck Stretch",
# #     "Half moon pose",
# #     "Gate yoga pose",
# #     "Upward Salute",
# #     "Table to toe pose",
# #     "Thread the needle pose",
# #     "Standing Quad Stretch Pose",
# #     "Tadasana with Lateral Bend",
# #     "Revolved Side Angle Pose",
# #     "Pyramid Pose",
# #     "Goddess Pose",
# #     "Reverse Warrior",
# #     "Standing Bow Pose",
# #     "Standing Figure Four Pose",
# #     "Revolved Triangle Pose",
# #     "Standing Spinal Twist Pose",
# #     "Lunging Calf Stretch",
# #     "Standing Bent Over Calf Strength",
# #     "Half-Split Strength",
# #     "frog pose",
# #     "Camel pose",
# #     "Cat cow pose",
# #     "Cow Face pose",
# #     "extended side angle pose",
# #     "Garland pose",
# #     "half lord of the fishes pose",
# #     "Hero pose",
# #     "Assisted side bend",
# #     "Finger Up and Down Strecth",
# #     "Salutation Seal",
# #     "Head to knee Forward Bend",
# #     "The big toe pose",
# #     "Sage Marichi",
# #     "Hald forward Bend",
# #     "Plow pose",
# #     "Bow pose",
# #     "Cactus pose"
# # ]


# # รายชื่อท่าโยคะทั้งหมดตามลำดับ
# POSE_NAMES = [
#     "Triangle Pose (Trikonasana)",
#     "Utkatasana (The Chair Pose)",
#     "Sukhasana (Easy Pose)",
#     "Butterfly Pose_Bound Angle Pose (Baddha Konasana)",
#     "Mountain Pose (Tadasana)",
#     "Chair Twist Pose",
#     "lizard pose",
#     "crescent lunge twist yoga pose",
#     "Revolved Janosha shansana",
#     "Bird Dog Pose",
#     "Side Neck Stretch",
#     "Gate yoga pose",
#     "Upward Salute",
#     "Standing Quad Stretch Pose",
#     "Tadasana with Lateral Bend",
#     "Revolved Side Angle Pose",
#     "Pyramid Pose",
#     "Goddess Pose",
#     "Reverse Warrior",
#     "Standing Bow Pose",
#     "Standing Figure Four Pose",
#     "Revolved Triangle Pose",
#     "Standing Spinal Twist Pose",
#     "Lunging Calf Stretch",
#     "Standing Bent Over Calf Strength",
#     "Half-Split Strength",
#     "extended side angle pose",
#     "Hero pose",
#     "Assisted side bend",
#     "Salutation Seal",
#     "Head to knee Forward Bend",
#     "The big toe pose",
#     "Hald forward Bend",
#     "Plow pose",
#     "Bow pose",
#     "Cactus pose",
#     "warrior2",
#     "warrior 1",
#     "tree",
#     "Boat_Pose_or_Paripurna_Navasana_",
#     "Bridge_Pose_or_Setu_Bandha_Sarvangasana_",
#     "Camel_Pose_or_Ustrasana_",
#     "Cat_Cow_Pose_or_Marjaryasana_",
#     "Cobra_Pose_or_Bhujangasana_",
#     "Corpse_Pose_or_Savasana_",
#     "Half_Lord_of_the_Fishes_Pose_or_Ardha_Matsyendrasana_",
#     "Half_Moon_Pose_or_Ardha_Chandrasana_",
#     "Lord_of_the_Dance_Pose_or_Natarajasana_",
#     "Low_Lunge_pose_or_Anjaneyasana_",
#     "Rajakapotasana",
#     "Side_Plank_Pose_or_Vasisthasana_",
#     "Side-Reclining_Leg_Lift_pose_or_Anantasana_",
#     "Warrior3"
# ]

# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         # รับข้อมูลจาก request
#         data = request.json
#         keypoints = data.get('keypoints', [])
#         allowed_poses = data.get('allowedPoses', [])
        
#         print(f"รับคำขอการทำนายท่าโยคะ")
#         print(f"ท่าที่อนุญาต: {allowed_poses}")
        
#         # แปลงเป็น numpy array ตามรูปแบบที่โมเดลต้องการ
#         keypoints = np.array(keypoints).reshape(1, 33, 3)
        
#         # ใช้โมเดลทำนายท่าทั้งหมด
#         prediction = model.predict(keypoints)
        
#         # ถ้าไม่มีการระบุท่าที่อนุญาต
#         if not allowed_poses or len(allowed_poses) == 0:
#             # ทำนายแบบปกติ (ทุกท่า)
#             predicted_class = np.argmax(prediction[0])
#             confidence = float(np.max(prediction[0]))
#             predicted_pose = POSE_NAMES[predicted_class]
#         else:
#             # กรองผลลัพธ์เฉพาะท่าที่อยู่ในรายการที่อนุญาต
#             filtered_scores = []
            
#             for idx, pose_name in enumerate(POSE_NAMES):
#                 # ตรวจสอบว่าท่านี้อยู่ในรายการที่อนุญาตหรือไม่
#                 for allowed_pose in allowed_poses:
#                     # พยายามทำการจับคู่แบบยืดหยุ่น - เช่นถ้าชื่อคล้ายกันหรือมีอยู่ในกันและกัน
#                     if (allowed_pose.lower() in pose_name.lower() or 
#                         pose_name.lower() in allowed_pose.lower()):
#                         score = prediction[0][idx]
#                         filtered_scores.append((pose_name, score, idx))
#                         break
            
#             # ถ้ามีท่าที่อนุญาตและตรงกับโมเดล
#             if filtered_scores:
#                 # เลือกท่าที่มีคะแนนสูงสุดในบรรดาท่าที่อนุญาต
#                 predicted_pose, confidence, class_idx = max(filtered_scores, key=lambda x: x[1])
#                 print(f"ท่าที่ทำนายได้: {predicted_pose} (คะแนน: {confidence:.2f})")
#             else:
#                 # ถ้าไม่มีท่าตรงกัน ใช้ค่าเริ่มต้น
#                 predicted_pose = "Unknown Pose"
#                 confidence = 0.0
#                 class_idx = -1
#                 print("ไม่พบท่าที่ตรงกับรายการที่อนุญาต")
        
#         # ส่งผลการทำนายกลับ
#         return jsonify({
#             "predicted_pose": predicted_pose,
#             "confidence": confidence,
#             "class_idx": int(class_idx) if 'class_idx' in locals() else -1
#         }), 200
            
#     except Exception as e:
#         print(f"Error in prediction: {str(e)}")
#         return jsonify({"error": str(e)}), 500

# # @app.route('/predict', methods=['POST'])
# # def predict():
# #     try:
# #         # รับ keypoints จาก request
# #         data = request.json
# #         keypoints = data.get('keypoints', [])
        
# #         if not keypoints:
# #             return jsonify({"error": "No keypoints provided"}), 400
            
# #         # แปลงเป็น numpy array และ reshape ให้เหมาะกับ LSTM
# #         keypoints = np.array(keypoints).reshape(1, 33, 3)
        
# #         # Predict
# #         prediction = model.predict(keypoints)
# #         predicted_class = np.argmax(prediction[0])
# #         confidence = float(np.max(prediction[0]))
        
# #         # หา top 3 predictions
# #         top_3_indices = np.argsort(prediction[0])[-3:][::-1]
# #         top_3_predictions = [
# #             {
# #                 "pose": POSE_NAMES[idx],
# #                 "confidence": float(prediction[0][idx])
# #             } for idx in top_3_indices
# #         ]
        
# #         return jsonify({
# #             "predicted_pose": POSE_NAMES[predicted_class],
# #             "confidence": confidence,
# #             "top_3_predictions": top_3_predictions
# #         }), 200
            
# #     except Exception as e:
# #         print(f"Error in prediction: {str(e)}")
# #         return jsonify({"error": str(e)}), 500

# if __name__ == '__main__':
#     print(f"Model loaded successfully. Ready to predict {len(POSE_NAMES)} yoga poses.")
#     app.run(host='0.0.0.0', port=5000, debug=True)

import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import logging

app = Flask(__name__)
CORS(app)

# ตั้งค่า logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# โหลดโมเดล
MODEL_PATH = 'uploads/yoga_pose_model_best_folddd28268.h5'
# MODEL_PATH = 'uploads/yoga_pose_model_best_folddd6368.h5'
model = tf.keras.models.load_model(MODEL_PATH)

# โหลดข้อมูลค่าเฉลี่ยมุมของแต่ละท่าจาก CSV
ANGLE_DATA_PATH = 'uploads/yoga_pose_average_angles_nameddddd.csv'
angle_df = pd.read_csv(ANGLE_DATA_PATH)
logger.info(f"โหลดข้อมูลมุมทั้งหมด {len(angle_df)} ท่า")

# POSE_NAMES = [
#     "Triangle Pose (Trikonasana)",
#     "Utkatasana (The Chair Pose)",
#     "Sukhasana (Easy Pose)",
#     "Butterfly Pose_Bound Angle Pose (Baddha Konasana)",
#     "Mountain Pose (Tadasana)",
#     "Chair Twist Pose",
#     "lizard pose",
#     "crescent lunge twist yoga pose",
#     "Revolved Janosha shansana",
#     "Bird Dog Pose",
#     "Side Neck Stretch",
#     "Gate yoga pose",
#     "Upward Salute",
#     "Standing Quad Stretch Pose",
#     "Tadasana with Lateral Bend",
#     "Revolved Side Angle Pose",
#     "Pyramid Pose",
#     "Goddess Pose",
#     "Reverse Warrior",
#     "Standing Bow Pose",
#     "Standing Figure Four Pose",
#     "Revolved Triangle Pose",
#     "Standing Spinal Twist Pose",
#     "Lunging Calf Stretch",
#     "Standing Bent Over Calf Strength",
#     "Half-Split Strength",
#     "extended side angle pose",
#     "Hero pose",
#     "Assisted side bend",
#     "Salutation Seal",
#     "Head to knee Forward Bend",
#     "The big toe pose",
#     "Hald forward Bend",
#     "Plow pose",
#     "Bow pose",
#     "Cactus pose",
#     "warrior2",
#     "warrior 1",
#     "Tree Pose",
#     "Boat_Pose_or_Paripurna_Navasana_",
#     "Bridge_Pose_or_Setu_Bandha_Sarvangasana_",
#     "Camel_Pose_or_Ustrasana_",
#     "Cat_Cow_Pose_or_Marjaryasana_",
#     "Cobra_Pose_or_Bhujangasana_",
#     "Corpse_Pose_or_Savasana_",
#     "Half_Lord_of_the_Fishes_Pose_or_Ardha_Matsyendrasana_",
#     "Half_Moon_Pose_or_Ardha_Chandrasana_",
#     "Lord_of_the_Dance_Pose_or_Natarajasana_",
#     "Low_Lunge_pose_or_Anjaneyasana_",
#     "Rajakapotasana",
#     "Side_Plank_Pose_or_Vasisthasana_",
#     "Side-Reclining_Leg_Lift_pose_or_Anantasana_",
#     "Warrior3"
# ]

POSE_NAMES = [
    "Triangle Pose",                  # จากเดิม: Triangle Pose (Trikonasana)
    "The Chair Pose",                 # จากเดิม: Utkatasana (The Chair Pose)
    "Easy Pose",                      # จากเดิม: Sukhasana (Easy Pose)
    "Butterfly Pose",                 # จากเดิม: Butterfly Pose_Bound Angle Pose (Baddha Konasana)
    "Mountain Pose",                  # จากเดิม: Mountain Pose (Tadasana)
    "Chair Twist Pose",               # คงเดิม
    "Lizard Pose",                    # จากเดิม: lizard pose
    "Crescent Lunge Twist Pose",      # จากเดิม: crescent lunge twist yoga pose
    "Revolved Head To Knee Pose",     # จากเดิม: Revolved Janosha shansana
    "Bird Dog Pose",                  # คงเดิม
    "Side Neck Stretch Pose",         # จากเดิม: Side Neck Stretch
    "Gate Pose",                      # จากเดิม: Gate yoga pose
    "Upward Salute Pose",             # จากเดิม: Upward Salute
    "Standing Quad Stretch Pose",     # คงเดิม
    "Upward Salute Side Bend Pose",   # จากเดิม: Tadasana with Lateral Bend
    "Revolved Side Angle Pose",       # คงเดิม
    "Pyramid Pose",                   # คงเดิม
    "Goddess Pose",                   # คงเดิม
    "Reverse Warrior Pose",           # จากเดิม: Reverse Warrior
    "Standing Bow Pose",              # คงเดิม
    "Standing Figure Four Pose",      # คงเดิม
    "Revolved Triangle Pose",         # คงเดิม
    "Standing Spinal Twist Pose",     # คงเดิม
    "Lunging Calf Stretch Pose",      # จากเดิม: Lunging Calf Stretch
    "Standing Bent Over Calf Strength Pose",  # จากเดิม: Standing Bent Over Calf Strength
    "Half Split Pose",                # จากเดิม: Half-Split Strength
    "Extended Side Angle Pose",       # จากเดิม: extended side angle pose
    "Hero Pose",                      # จากเดิม: Hero pose
    "Assisted Side Bend Pose",        # จากเดิม: Assisted side bend
    "Salutation Seal Pose",           # จากเดิม: Salutation Seal
    "Head To Knee Forward Bend Pose", # จากเดิม: Head to knee Forward Bend
    "Big Toe Pose",                   # จากเดิม: The big toe pose
    "Half Forward Bend Pose",         # จากเดิม: Hald forward Bend
    "Plow Pose",                      # จากเดิม: Plow pose
    "Bow Pose",                       # จากเดิม: Bow pose
    "Cactus Pose",                    # จากเดิม: Cactus pose
    "Warrior 2 Pose",                 # จากเดิม: warrior2
    "Warrior 1 Pose",                 # จากเดิม: warrior 1
    "Tree Pose",                      # จากเดิม: tree
    "Boat Pose",                      # จากเดิม: Boat_Pose_or_Paripurna_Navasana_
    "Bridge Pose",                    # จากเดิม: Bridge_Pose_or_Setu_Bandha_Sarvangasana_
    "Camel Pose",                     # จากเดิม: Camel_Pose_or_Ustrasana_
    "Cat Cow Pose",                   # จากเดิม: Cat_Cow_Pose_or_Marjaryasana_
    "Cobra Pose",                     # จากเดิม: Cobra_Pose_or_Bhujangasana_
    "Corpse Pose",                    # จากเดิม: Corpse_Pose_or_Savasana_
    "Half Lord Of The Fishes Pose",   # จากเดิม: Half_Lord_of_the_Fishes_Pose_or_Ardha_Matsyendrasana_
    "Half Moon Pose",                 # จากเดิม: Half_Moon_Pose_or_Ardha_Chandrasana_
    "Dancer Pose",                    # จากเดิม: Lord_of_the_Dance_Pose_or_Natarajasana_
    "Low Lunge Pose",                 # จากเดิม: Low_Lunge_pose_or_Anjaneyasana_
    "King Pigeon Pose",               # จากเดิม: Rajakapotasana
    "Side Plank Pose",                # จากเดิม: Side_Plank_Pose_or_Vasisthasana_
    "Side Reclining Leg Lift Pose",   # จากเดิม: Side-Reclining_Leg_Lift_pose_or_Anantasana_
    "Warrior 3 Pose"                  # จากเดิม: Warrior3
]

# เตรียมข้อมูลมุมของแต่ละท่าไว้ใช้อ้างอิง
pose_angle_references = {}
for _, row in angle_df.iterrows():
    pose_name = row['pose_name']
    angle_data = {
        'left_shoulder_angle': row['left_shoulder_angle'],
        'right_shoulder_angle': row['right_shoulder_angle'],
        'left_elbow_angle': row['left_elbow_angle'],
        'right_elbow_angle': row['right_elbow_angle'],
        'left_hip_angle': row['left_hip_angle'],
        'right_hip_angle': row['right_hip_angle'],
        'left_knee_angle': row['left_knee_angle'],
        'right_knee_angle': row['right_knee_angle']
    }
    pose_angle_references[pose_name] = angle_data

def calculate_angle_similarity(detected_angles, reference_angles):
    """
    คำนวณความคล้ายคลึงของมุมระหว่างมุมที่ตรวจจับได้กับมุมอ้างอิงของท่า
    
    return: คะแนนความคล้ายคลึง (0-100)
    """
    angle_columns = [
        'left_shoulder_angle', 'right_shoulder_angle',
        'left_elbow_angle', 'right_elbow_angle', 
        'left_hip_angle', 'right_hip_angle',
        'left_knee_angle', 'right_knee_angle'
    ]
    
    # น้ำหนักของแต่ละมุม (ปรับตามความสำคัญ)
    weights = {
        'left_shoulder_angle': 1.2, 'right_shoulder_angle': 1.2,
        'left_elbow_angle': 1.0, 'right_elbow_angle': 1.0,
        'left_hip_angle': 1.5, 'right_hip_angle': 1.5,
        'left_knee_angle': 1.3, 'right_knee_angle': 1.3
    }
    
    total_score = 0
    max_possible_score = 0
    
    for angle_name in angle_columns:
        if angle_name in detected_angles and angle_name in reference_angles:
            detected = detected_angles[angle_name]
            reference = reference_angles[angle_name]
            weight = weights.get(angle_name, 1.0)
            
            # คำนวณความแตกต่างเป็นเปอร์เซ็นต์ (หากต่างกัน 45 องศา = 0%)
            max_diff = 45.0  # องศาสูงสุดที่ยอมรับได้
            diff = min(abs(detected - reference), max_diff)
            angle_score = (1 - diff / max_diff) * 100 * weight
            
            total_score += angle_score
            max_possible_score += 100 * weight
    
    # คำนวณคะแนนรวม (เป็นเปอร์เซ็นต์)
    if max_possible_score > 0:
        similarity = (total_score / max_possible_score) * 100
    else:
        similarity = 0
    
    return similarity

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # รับข้อมูลจาก request
        data = request.json
        keypoints = data.get('keypoints', [])
        joint_angles = data.get('joint_angles', {})
        allowed_poses = data.get('allowedPoses', [])
        
        # แปลงเป็น numpy array ตามรูปแบบที่โมเดลต้องการ
        keypoints = np.array(keypoints).reshape(1, 33, 3)
        
        # ใช้โมเดลทำนายชื่อท่า
        prediction = model.predict(keypoints)
        
        # ถ้าไม่มีการระบุท่าที่อนุญาต
        if not allowed_poses or len(allowed_poses) == 0:
            # ทำนายแบบปกติ (ทุกท่า)
            predicted_class = np.argmax(prediction[0])
            confidence = float(np.max(prediction[0]))
            predicted_pose = POSE_NAMES[predicted_class]
            
            # คำนวณความคล้ายคลึงของมุม
            angle_similarity = 0.0
            if joint_angles and predicted_pose in pose_angle_references:
                reference_angles = pose_angle_references[predicted_pose]
                angle_similarity = calculate_angle_similarity(joint_angles, reference_angles)
            
            # ส่งผลการทำนายกลับ
            return jsonify({
                "predicted_pose": predicted_pose,
                "confidence": confidence,
                "angle_similarity": angle_similarity,
                "class_idx": int(predicted_class)
            }), 200
        else:
            # กรองผลลัพธ์เฉพาะท่าที่อยู่ในรายการที่อนุญาต
            filtered_scores = []
            
            for idx, pose_name in enumerate(POSE_NAMES):
                # ตรวจสอบว่าท่านี้อยู่ในรายการที่อนุญาตหรือไม่
                for allowed_pose in allowed_poses:
                    # พยายามทำการจับคู่แบบยืดหยุ่น
                    if (allowed_pose.lower() in pose_name.lower() or 
                        pose_name.lower() in allowed_pose.lower()):
                        
                        # คะแนนจากโมเดล
                        model_score = prediction[0][idx]
                        
                        # คำนวณความคล้ายคลึงของมุม
                        angle_similarity = 0.0
                        if joint_angles and pose_name in pose_angle_references:
                            reference_angles = pose_angle_references[pose_name]
                            angle_similarity = calculate_angle_similarity(joint_angles, reference_angles)
                        
                        filtered_scores.append({
                            'pose_name': pose_name,
                            'model_score': float(model_score),
                            'angle_similarity': float(angle_similarity),
                            'class_idx': idx
                        })
                        break
            
            # ถ้ามีท่าที่อนุญาตและตรงกับโมเดล
            if filtered_scores:
                # เลือกท่าที่มีคะแนนรวมสูงสุดจากโมเดล
                best_match = max(filtered_scores, key=lambda x: x['model_score'])
                
                return jsonify({
                    "predicted_pose": best_match['pose_name'],
                    "confidence": best_match['model_score'],
                    "angle_similarity": best_match['angle_similarity'],
                    "class_idx": int(best_match['class_idx'])
                }), 200
            else:
                # ถ้าไม่มีท่าตรงกัน ใช้ค่าเริ่มต้น
                return jsonify({
                    "predicted_pose": "Unknown Pose",
                    "confidence": 0.0,
                    "angle_similarity": 0.0,
                    "class_idx": -1
                }), 200
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    logger.info(f"โมเดลโหลดเรียบร้อยแล้ว พร้อมทำนาย {len(POSE_NAMES)} ท่า")
    app.run(host='0.0.0.0', port=5000, debug=True)