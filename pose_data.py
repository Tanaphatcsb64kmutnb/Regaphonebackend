# กำหนดข้อมูลมุมมาตรฐานของแต่ละท่า
POSE_ANGLES = {
    "Cat Cow Pose": {
        'right_hip': {'min': 70, 'max': 110},
        'left_hip': {'min': 70, 'max': 110},
        'right_knee': {'min': 80, 'max': 100},
        'left_knee': {'min': 80, 'max': 100},
        'right_shoulder': {'min': 80, 'max': 100},
        'left_shoulder': {'min': 80, 'max': 100}
    },
    
    "Camel Pose": {
        'right_hip': {'min': 160, 'max': 200},
        'left_hip': {'min': 160, 'max': 200},
        'right_knee': {'min': 80, 'max': 100},
        'left_knee': {'min': 80, 'max': 100},
        'right_shoulder': {'min': 30, 'max': 60},
        'left_shoulder': {'min': 30, 'max': 60}
    }
}

# ข้อความ feedback สำหรับแต่ละท่า
POSE_FEEDBACK = {
    "Cat Cow Pose": {
        'right_hip': {
            'too_low': "ยกสะโพกขวาขึ้นอีกนิด",
            'too_high': "ลดสะโพกขวาลงเล็กน้อย"
        },
        'left_hip': {
            'too_low': "ยกสะโพกซ้ายขึ้นอีกนิด",
            'too_high': "ลดสะโพกซ้ายลงเล็กน้อย"
        },
        # เพิ่ม feedback สำหรับข้อต่ออื่นๆ
    },
    
    "Camel Pose": {
        'right_shoulder': {
            'too_low': "ยกไหล่ขวาขึ้นอีกนิด",
            'too_high': "ลดไหล่ขวาลงเล็กน้อย"
        },
        'left_shoulder': {
            'too_low': "ยกไหล่ซ้ายขึ้นอีกนิด",
            'too_high': "ลดไหล่ซ้ายลงเล็กน้อย"
        },
        # เพิ่ม feedback สำหรับข้อต่ออื่นๆ
    }
}

def get_pose_score(pose_name, angles):
    """คำนวณคะแนนท่าทาง"""
    if pose_name not in POSE_ANGLES:
        return 0, []
        
    target_angles = POSE_ANGLES[pose_name]
    feedback = []
    score = 0
    total_points = len(target_angles)
    
    for joint, angle in angles.items():
        if joint in target_angles:
            target = target_angles[joint]
            if target['min'] <= angle <= target['max']:
                score += 1
            else:
                # เพิ่ม feedback
                if angle < target['min']:
                    feedback.append(POSE_FEEDBACK[pose_name][joint]['too_low'])
                else:
                    feedback.append(POSE_FEEDBACK[pose_name][joint]['too_high'])
                    
    return (score / total_points) * 100, feedback