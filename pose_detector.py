import mediapipe as mp
import numpy as np
import cv2

class PoseDetector:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=1
        )
        
    def detect_pose(self, image):
        """ตรวจจับท่าทางและคำนวณมุมต่างๆ"""
        # แปลงเป็น RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)
        
        if not results.pose_landmarks:
            return None, None
            
        # ดึงจุดสำคัญ
        landmarks = results.pose_landmarks.landmark
        angles = self.calculate_angles(landmarks)
        
        return angles, landmarks

    def calculate_angles(self, landmarks):
        """คำนวณมุมของข้อต่อต่างๆ"""
        angles = {
            # มุมข้อศอกขวา
            'right_elbow': self._get_angle(
                (landmarks[12].x, landmarks[12].y), # shoulder
                (landmarks[14].x, landmarks[14].y), # elbow
                (landmarks[16].x, landmarks[16].y)  # wrist
            ),
            
            # มุมข้อศอกซ้าย
            'left_elbow': self._get_angle(
                (landmarks[11].x, landmarks[11].y),
                (landmarks[13].x, landmarks[13].y),
                (landmarks[15].x, landmarks[15].y)
            ),
            
            # มุมไหล่ขวา
            'right_shoulder': self._get_angle(
                (landmarks[14].x, landmarks[14].y), # elbow
                (landmarks[12].x, landmarks[12].y), # shoulder
                (landmarks[24].x, landmarks[24].y)  # hip
            ),
            
            # มุมไหล่ซ้าย
            'left_shoulder': self._get_angle(
                (landmarks[13].x, landmarks[13].y),
                (landmarks[11].x, landmarks[11].y),
                (landmarks[23].x, landmarks[23].y)
            ),
            
            # มุมสะโพกขวา
            'right_hip': self._get_angle(
                (landmarks[12].x, landmarks[12].y), # shoulder
                (landmarks[24].x, landmarks[24].y), # hip
                (landmarks[26].x, landmarks[26].y)  # knee
            ),
            
            # มุมสะโพกซ้าย
            'left_hip': self._get_angle(
                (landmarks[11].x, landmarks[11].y),
                (landmarks[23].x, landmarks[23].y),
                (landmarks[25].x, landmarks[25].y)
            ),
            
            # มุมเข่าขวา
            'right_knee': self._get_angle(
                (landmarks[24].x, landmarks[24].y), # hip
                (landmarks[26].x, landmarks[26].y), # knee
                (landmarks[28].x, landmarks[28].y)  # ankle
            ),
            
            # มุมเข่าซ้าย
            'left_knee': self._get_angle(
                (landmarks[23].x, landmarks[23].y),
                (landmarks[25].x, landmarks[25].y),
                (landmarks[27].x, landmarks[27].y)
            )
        }
        return angles

    def _get_angle(self, p1, p2, p3):
        """คำนวณมุมระหว่าง 3 จุด"""
        radians = np.arctan2(p3[1] - p2[1], p3[0] - p2[0]) - \
                 np.arctan2(p1[1] - p2[1], p1[0] - p2[0])
        angle = np.abs(radians * 180.0 / np.pi)
        
        if angle > 180.0:
            angle = 360 - angle
            
        return angle