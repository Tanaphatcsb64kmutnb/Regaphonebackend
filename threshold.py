import numpy as np
import pandas as pd
import csv
import os
import argparse
from collections import defaultdict

def main():
    parser = argparse.ArgumentParser(description='Calculate angle thresholds for yoga poses')
    parser.add_argument('--input', type=str, default='yoga_pose_data.csv', help='Input CSV file with pose data')
    parser.add_argument('--output', type=str, default='pose_thresholds.csv', help='Output CSV file for thresholds')
    args = parser.parse_args()
    
    print(f"Processing {args.input} to generate angle thresholds...")
    
    # ตรวจสอบไฟล์ข้อมูล
    if not os.path.exists(args.input):
        print(f"Error: Input file {args.input} not found.")
        return
    
    # โหลดข้อมูลท่าโยคะ
    df = pd.read_csv(args.input)
    
    # ตรวจสอบคอลัมน์
    required_columns = ['pose_name', 'keypoints']
    if not all(col in df.columns for col in required_columns):
        print(f"Error: Missing required columns. Required: {required_columns}")
        return
    
    # เก็บข้อมูลมุมข้อต่อตามท่าต่างๆ
    pose_angles = defaultdict(lambda: defaultdict(list))
    
    # วนลูปตามข้อมูลและคำนวณมุมข้อต่อ
    for _, row in df.iterrows():
        pose_name = row['pose_name']
        keypoints = eval(row['keypoints'])  # ต้องเก็บในรูปแบบที่ eval ได้
        
        # แปลง keypoints เป็น numpy array ขนาด (33, 3)
        keypoints_array = np.array(keypoints).reshape(33, 3)
        
        # คำนวณมุมข้อต่อจาก keypoints
        joint_angles = calculate_joint_angles(keypoints_array)
        
        # เก็บมุมข้อต่อตามท่า
        for joint, angle in joint_angles.items():
            pose_angles[pose_name][joint].append(angle)
    
    # คำนวณค่าเฉลี่ยและค่าเบี่ยงเบนมาตรฐานของมุมแต่ละข้อสำหรับแต่ละท่า
    pose_thresholds = {}
    
    for pose_name, joints in pose_angles.items():
        pose_thresholds[pose_name] = {}
        
        for joint, angles in joints.items():
            angles_array = np.array(angles)
            mean_angle = np.mean(angles_array)
            std_angle = np.std(angles_array)
            
            # เก็บค่าเฉลี่ยและกำหนด tolerance จากค่าเบี่ยงเบนมาตรฐาน
            pose_thresholds[pose_name][joint] = mean_angle
            pose_thresholds[pose_name][f"{joint}_tolerance"] = max(15.0, std_angle * 2.0)  # อย่างน้อย 15 องศา
    
    # บันทึกค่า threshold ลง CSV
    with open(args.output, 'w', newline='') as csvfile:
        # หาคอลัมน์ทั้งหมด
        all_columns = ['pose_name']
        for pose_thresholds in pose_thresholds.values():
            for joint in pose_thresholds.keys():
                if joint not in all_columns:
                    all_columns.append(joint)
        
        writer = csv.DictWriter(csvfile, fieldnames=all_columns)
        writer.writeheader()
        
        for pose_name, thresholds in pose_thresholds.items():
            row = {'pose_name': pose_name}
            row.update(thresholds)
            writer.writerow(row)
    
    print(f"Thresholds for {len(pose_thresholds)} poses have been saved to {args.output}")

def calculate_joint_angles(landmarks):
    """
    คำนวณมุมข้อต่อจาก landmarks
    landmarks: numpy array ขนาด (33, 3) ตาม MediaPipe Pose format
    """
    angles = {}
    
    # คำนวณมุมข้อศอกขวา (RIGHT_ELBOW)
    angles["RIGHT_ELBOW"] = calculate_angle(
        landmarks[11],  # RIGHT_SHOULDER
        landmarks[13],  # RIGHT_ELBOW
        landmarks[15]   # RIGHT_WRIST
    )
    
    # คำนวณมุมข้อศอกซ้าย (LEFT_ELBOW)
    angles["LEFT_ELBOW"] = calculate_angle(
        landmarks[12],  # LEFT_SHOULDER
        landmarks[14],  # LEFT_ELBOW
        landmarks[16]   # LEFT_WRIST
    )
    
    # คำนวณมุมข้อไหล่ขวา (RIGHT_SHOULDER)
    angles["RIGHT_SHOULDER"] = calculate_angle(
        landmarks[23],  # RIGHT_HIP
        landmarks[11],  # RIGHT_SHOULDER
        landmarks[13]   # RIGHT_ELBOW
    )
    
    # คำนวณมุมข้อไหล่ซ้าย (LEFT_SHOULDER)
    angles["LEFT_SHOULDER"] = calculate_angle(
        landmarks[24],  # LEFT_HIP
        landmarks[12],  # LEFT_SHOULDER
        landmarks[14]   # LEFT_ELBOW
    )
    
    # คำนวณมุมข้อสะโพกขวา (RIGHT_HIP)
    angles["RIGHT_HIP"] = calculate_angle(
        landmarks[11],  # RIGHT_SHOULDER
        landmarks[23],  # RIGHT_HIP
        landmarks[25]   # RIGHT_KNEE
    )
    
    # คำนวณมุมข้อสะโพกซ้าย (LEFT_HIP)
    angles["LEFT_HIP"] = calculate_angle(
        landmarks[12],  # LEFT_SHOULDER
        landmarks[24],  # LEFT_HIP
        landmarks[26]   # LEFT_KNEE
    )
    
    # คำนวณมุมข้อเข่าขวา (RIGHT_KNEE)
    angles["RIGHT_KNEE"] = calculate_angle(
        landmarks[23],  # RIGHT_HIP
        landmarks[25],  # RIGHT_KNEE
        landmarks[27]   # RIGHT_ANKLE
    )
    
    # คำนวณมุมข้อเข่าซ้าย (LEFT_KNEE)
    angles["LEFT_KNEE"] = calculate_angle(
        landmarks[24],  # LEFT_HIP
        landmarks[26],  # LEFT_KNEE
        landmarks[28]   # LEFT_ANKLE
    )
    
    # คำนวณมุมแนวลำตัว (TORSO_ANGLE)
    mid_shoulder = (landmarks[11] + landmarks[12]) / 2  # จุดกึ่งกลางไหล่
    mid_hip = (landmarks[23] + landmarks[24]) / 2       # จุดกึ่งกลางสะโพก
    
    # มุมระหว่างแนวตั้งกับแนวลำตัว
    v1 = np.array([0, -1, 0])  # แนวตั้ง
    v2 = mid_shoulder - mid_hip  # แนวลำตัว
    
    angles["TORSO_ANGLE"] = calculate_angle_vectors(v1, v2)
    
    return angles

def calculate_angle(p1, p2, p3):
    """
    คำนวณมุมระหว่างสามจุด p1-p2-p3 ในหน่วยองศา
    """
    v1 = p1 - p2
    v2 = p3 - p2
    
    return calculate_angle_vectors(v1, v2)

def calculate_angle_vectors(v1, v2):
    """
    คำนวณมุมระหว่างสองเวกเตอร์ในหน่วยองศา
    """
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    cos_angle = np.clip(cos_angle, -1.0, 1.0)  # ป้องกันค่า NaN
    angle = np.arccos(cos_angle) * 180.0 / np.pi
    
    return angle

if __name__ == "__main__":
    main()