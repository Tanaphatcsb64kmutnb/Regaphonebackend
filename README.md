# Yoga Pose Evaluation System

A Flask-based backend system for real-time yoga pose evaluation and scoring.

## 🚀 Project Overview

This project uses a trained deep learning model to analyze user poses via keypoints and joint positions, verifying their accuracy based on yoga posture standards. Users are rewarded with points when they perform a pose correctly.

---

## 🧠 How It Works

1. **Backend Technology**  
   - Developed using **Python Flask** framework.
   - The core model is located at:  
     `my_flask_app/uploads/yoga_pose_model_best_folddd28268.h5`.

2. **Yoga Pose Prediction**  
   - The model receives input in the form of keypoints and joint positions.
   - It predicts whether the user is performing the yoga pose correctly or not.

3. **Score Calculation**  
   - If the pose is predicted to be **correct**, the system calculates a **yoga score**.
   - Points are accumulated and stored for the user profile.

4. **System Architecture**  
   - You can find the overall backend system design here:  
     `Rega-Project/assets/systemarchitecturebackend.png`

5. **Deployment**  
   - The backend is deployed using [**Railway**](https://railway.app).

---

## 📁 Project Structure (Simplified)

