# 🧘‍♀️ Yoga Pose Evaluation System

A Flask-based backend system for real-time yoga pose evaluation and scoring.

---

## 🚀 Project Overview

This project utilizes a trained deep learning model to analyze user yoga poses based on body keypoints and joint positions.  
If the pose is performed correctly, the system awards points to the user, which are accumulated over time.

---

## 🧠 How It Works

### 1. Backend Technology  
- Developed using the **Python Flask** framework.  
- The pose evaluation model is located at:  
  `uploads/yoga_pose_model_best_folddd28268.h5`  

<img src="assets/flask.png" alt="Flask Logo" width="100" />

---

### 2. Yoga Pose Prediction  
- The system receives keypoints and joint position data as input.  
- The model predicts whether the user is performing the yoga pose correctly.

---

### 3. Score Calculation  
- When a pose is correctly performed, the system:
  - Calculates a **yoga score**
  - Adds it to the user's total accumulated points

---

### 4. System Architecture  
- The overall backend design is illustrated below:  
  <img src="assets/systemarchitecturebackend.png" width="1000" alt="System Architecture"/>

---

### 5. Deployment  
- The backend is deployed via [**Railway**](https://railway.app), enabling seamless deployment and hosting.

<img src="assets/railway.png" alt="Railway Logo" width="130" />

---

## 📂 Project Structure

