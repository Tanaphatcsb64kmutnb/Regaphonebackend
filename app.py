import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import logging
import os

app = Flask(__name__)
CORS(app)

# ตั้งค่า logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# โหลดโมเดล
# MODEL_PATH = 'uploads/yoga_pose_model_best_folddd28268.h5'
# MODEL_PATH = os.environ.get('MODEL_PATH', 'uploads/yoga_pose_model_best_folddd28268.h5')
MODEL_PATH = "uploads/yoga_pose_model_best_folddd28268.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# โหลดข้อมูลค่าเฉลี่ยมุมของแต่ละท่าจาก CSV
# ANGLE_DATA_PATH = 'uploads/yoga_pose_average_angles_nameddddd.csv'
# ANGLE_DATA_PATH = os.environ.get('ANGLE_DATA_PATH', 'uploads/yoga_pose_average_angles_nameddddd.csv')
ANGLE_DATA_PATH = "uploads/yoga_pose_average_angles_nameddddd.csv"
angle_df = pd.read_csv(ANGLE_DATA_PATH)
logger.info(f"โหลดข้อมูลมุมทั้งหมด {len(angle_df)} ท่า")

# บังคับให้ TensorFlow ใช้งานเฉพาะ CPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# หรือตั้งค่าการใช้งาน GPU แบบมีเงื่อนไข
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
else:
    print("No GPU available, using CPU instead")


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

# import pandas as pd
# import numpy as np
# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import tensorflow as tf
# import logging

# app = Flask(__name__)
# CORS(app)

# # ตั้งค่า logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # โหลดโมเดล
# MODEL_PATH = 'uploads/yoga_pose_model_best_folddd28268.h5'
# model = tf.keras.models.load_model(MODEL_PATH)

# # โหลดข้อมูลค่าเฉลี่ยมุมของแต่ละท่าจาก CSV
# ANGLE_DATA_PATH = 'uploads/yoga_pose_average_angles_nameddddd.csv'
# angle_df = pd.read_csv(ANGLE_DATA_PATH)
# logger.info(f"โหลดข้อมูลมุมทั้งหมด {len(angle_df)} ท่า")

# # ชื่อท่าโยคะทั้งหมด (ใช้ชื่อที่ปรับให้สั้นและอ่านง่ายแล้ว)
# POSE_NAMES = [
#     "Triangle Pose",                  
#     "The Chair Pose",                 
#     "Easy Pose",                      
#     "Butterfly Pose",                 
#     "Mountain Pose",                  
#     "Chair Twist Pose",               
#     "Lizard Pose",                    
#     "Crescent Lunge Twist Pose",      
#     "Revolved Head To Knee Pose",     
#     "Bird Dog Pose",                  
#     "Side Neck Stretch Pose",         
#     "Gate Pose",                      
#     "Upward Salute Pose",             
#     "Standing Quad Stretch Pose",     
#     "Upward Salute Side Bend Pose",   
#     "Revolved Side Angle Pose",       
#     "Pyramid Pose",                   
#     "Goddess Pose",                   
#     "Reverse Warrior Pose",           
#     "Standing Bow Pose",              
#     "Standing Figure Four Pose",      
#     "Revolved Triangle Pose",         
#     "Standing Spinal Twist Pose",     
#     "Lunging Calf Stretch Pose",      
#     "Standing Bent Over Calf Strength Pose",  
#     "Half Split Pose",                
#     "Extended Side Angle Pose",       
#     "Hero Pose",                      
#     "Assisted Side Bend Pose",        
#     "Salutation Seal Pose",           
#     "Head To Knee Forward Bend Pose", 
#     "Big Toe Pose",                   
#     "Half Forward Bend Pose",         
#     "Plow Pose",                      
#     "Bow Pose",                       
#     "Cactus Pose",                    
#     "Warrior 2 Pose",                 
#     "Warrior 1 Pose",                 
#     "Tree Pose",                      
#     "Boat Pose",                      
#     "Bridge Pose",                    
#     "Camel Pose",                     
#     "Cat Cow Pose",                   
#     "Cobra Pose",                     
#     "Corpse Pose",                    
#     "Half Lord Of The Fishes Pose",   
#     "Half Moon Pose",                 
#     "Dancer Pose",                    
#     "Low Lunge Pose",                 
#     "King Pigeon Pose",               
#     "Side Plank Pose",                
#     "Side Reclining Leg Lift Pose",   
#     "Warrior 3 Pose"                  
# ]

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
    
    # ปรับเพิ่มค่า max_diff เป็น 60.0 เพื่อให้ยืดหยุ่นมากขึ้น
    max_diff = 60.0  # องศาสูงสุดที่ยอมรับได้
    
    for angle_name in angle_columns:
        if angle_name in detected_angles and angle_name in reference_angles:
            detected = detected_angles[angle_name]
            reference = reference_angles[angle_name]
            weight = weights.get(angle_name, 1.0)
            
            # คำนวณความแตกต่างเป็นเปอร์เซ็นต์
            diff = min(abs(detected - reference), max_diff)
            angle_score = (1 - diff / max_diff) * 100 * weight  # ไม่มี base_score
            
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
        
        # พิมพ์ข้อมูลที่ได้รับ
        logger.info(f"Received allowed poses: {allowed_poses}")
        logger.info(f"Received joint angles: {joint_angles}")
        
        # แปลงเป็น numpy array ตามรูปแบบที่โมเดลต้องการ
        keypoints = np.array(keypoints).reshape(1, 33, 3)
        
        # ใช้โมเดลทำนายชื่อท่า
        prediction = model.predict(keypoints)
        
        # ค่าเริ่มต้นสำหรับข้อมูลเพิ่มเติมที่จะส่งกลับ
        expected_pose = allowed_poses[0] if allowed_poses and len(allowed_poses) > 0 else None
        angle_discrepancies = {}

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
            
            # คำนวณความแตกต่างของมุมสำหรับฟีเจอร์ใหม่
            if expected_pose and joint_angles and expected_pose in pose_angle_references:
                reference_angles = pose_angle_references[expected_pose]
                
                # ปรับค่าความแตกต่างที่ยอมรับได้จาก 15 เป็น 30 องศา
                for angle_name, user_angle in joint_angles.items():
                    if angle_name in reference_angles:
                        ref_angle = reference_angles[angle_name]
                        # ถ้าความแตกต่างมีนัยสำคัญ (เพิ่มเป็น 30 องศา)
                        if abs(user_angle - ref_angle) > 30:
                            angle_discrepancies[angle_name] = {
                                'user_angle': user_angle,
                                'reference_angle': ref_angle,
                                'difference': abs(user_angle - ref_angle)
                            }
            
            logger.info(f"Predicted pose: {predicted_pose}, Angle similarity: {angle_similarity}")
            if angle_discrepancies:
                logger.info(f"Angle discrepancies: {angle_discrepancies}")
            
            # ส่งผลการทำนายกลับ
            return jsonify({
                "predicted_pose": predicted_pose,
                "confidence": confidence,
                "angle_similarity": angle_similarity,
                "class_idx": int(predicted_class),
                "expected_pose": expected_pose,
                "angle_discrepancies": angle_discrepancies
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
                predicted_pose = best_match['pose_name']
                confidence = best_match['model_score']
                angle_similarity = best_match['angle_similarity']
                class_idx = best_match['class_idx']
                
                # คำนวณความแตกต่างของมุมสำหรับฟีเจอร์ใหม่
                if expected_pose and joint_angles and expected_pose in pose_angle_references:
                    reference_angles = pose_angle_references[expected_pose]
                    
                    # ปรับค่าความแตกต่างที่ยอมรับได้จาก 15 เป็น 30 องศา
                    for angle_name, user_angle in joint_angles.items():
                        if angle_name in reference_angles:
                            ref_angle = reference_angles[angle_name]
                            # ถ้าความแตกต่างมีนัยสำคัญ (เพิ่มเป็น 40 องศา)
                            if abs(user_angle - ref_angle) > 40:
                                angle_discrepancies[angle_name] = {
                                    'user_angle': user_angle,
                                    'reference_angle': ref_angle,
                                    'difference': abs(user_angle - ref_angle)
                                }
                
                logger.info(f"Predicted pose: {predicted_pose}, Expected pose: {expected_pose}, Angle similarity: {angle_similarity}")
                logger.info(f"Angle discrepancies: {angle_discrepancies}")
                
                return jsonify({
                    "predicted_pose": predicted_pose,
                    "confidence": confidence,
                    "angle_similarity": angle_similarity,
                    "class_idx": int(class_idx),
                    "expected_pose": expected_pose,
                    "angle_discrepancies": angle_discrepancies
                }), 200
            else:
                # ถ้าไม่มีท่าตรงกัน ใช้ค่าเริ่มต้น
                return jsonify({
                    "predicted_pose": "Unknown Pose",
                    "confidence": 0.0,
                    "angle_similarity": 0.0,
                    "class_idx": -1,
                    "expected_pose": expected_pose,
                    "angle_discrepancies": {}
                }), 200
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

# if __name__ == '__main__':
#     logger.info(f"โมเดลโหลดเรียบร้อยแล้ว พร้อมทำนาย {len(POSE_NAMES)} ท่า")
#     app.run(host='0.0.0.0', port=5000, debug=True)

# ในส่วนล่างสุด
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)