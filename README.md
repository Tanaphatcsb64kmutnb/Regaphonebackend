🧘‍♀️ Yoga Pose Evaluation System
A real-time yoga pose evaluation and scoring system powered by Flask and deep learning.

🚀 Project Overview
This system uses a trained deep learning model to analyze yoga poses based on body keypoints and joint positions.
✅ When the user performs a correct pose, the system awards points — motivating consistent practice and improvement!

🧠 How It Works
1️⃣ Backend Technology
🧩 Built with Python Flask — lightweight, fast, and ideal for machine learning inference.
📁 Model location:
uploads/yoga_pose_model_best_folddd28268.h5

<img src="assets/flask.png" alt="Flask Logo" width="100"/>
2️⃣ Yoga Pose Prediction
📌 The system receives keypoints and joint position data from the user's pose.
🧠 The model evaluates if the pose matches the correct yoga form.

Joint Detection Example	Body Keypoints Example
<img src="assets/joint.png" width="300"/>	<img src="assets/body.png" width="300"/>

3️⃣ Score Calculation
💯 If the user performs the pose correctly:

A Yoga Score is calculated.

Points are added to the user's total score — promoting engagement over time.

4️⃣ System Architecture
🧱 The backend system is modular and scalable, with each part clearly separated for easy maintenance and upgrades.

<img src="assets/systemarchitecturebackend.png" width="1000" alt="System Architecture"/>
5️⃣ Deployment
🚀 Hosted on Railway for easy deployment and cloud hosting.

<img src="assets/railway.png" alt="Railway Logo" width="130"/>
✨ Features Recap
🧘‍♂️ Real-time yoga pose evaluation

🔍 Pose accuracy analysis using joint keypoints

🏅 Scoring system to motivate users

🌐 Deployed and accessible via Railway