# AI Football Analysis using Computer Vision ⚽🤖

This project focuses on analyzing football (soccer) match videos using **Computer Vision and Deep Learning** techniques. The goal is to automatically detect and track players, referees, and the ball, and extract meaningful insights such as player movement, speed, distance covered, and team-level statistics. The project demonstrates how modern AI models can be applied to real-world sports analytics problems.

---

## 📌 Project Overview

Football analysis traditionally requires manual effort and expert observation. This project automates that process using **YOLO-based object detection**, tracking algorithms, and geometric transformations. Given a football match video, the system processes each frame to identify objects of interest and computes useful performance metrics.

The project is designed to be modular, scalable, and reproducible, making it suitable for academic use, research, and practical experimentation in sports analytics.

---

## 🚀 Key Features

- **Object Detection**: Detects players, referees, goalkeepers, and the football using a trained YOLO model.
- **Object Tracking**: Maintains consistent IDs for detected objects across video frames.
- **Team Assignment**: Uses color-based clustering (K-Means) to assign players to teams based on jersey color.
- **Camera Motion Estimation**: Estimates camera movement using optical flow to improve tracking accuracy.
- **Perspective Transformation**: Converts pixel-based movement into real-world distances (meters).
- **Speed & Distance Estimation**: Calculates player speed and total distance covered during the match.
- **Clean Repository Design**: Trained models and datasets are excluded from version control.

---

## 🛠️ Technologies Used

- **Python**
- **PyTorch**
- **YOLO (Ultralytics)**
- **OpenCV**
- **NumPy**
- **Scikit-learn**
- **Roboflow** (for dataset management)

---

## 📂 Project Structure

project/
├── src/ # Core source code
├── scripts/ # Utility and helper scripts
├── configs/ # Configuration files
├── runs/ # YOLO outputs (ignored)
├── datasets/ # Roboflow datasets (ignored)
├── data.yaml # Dataset configuration
├── requirements.txt
├── .gitignore
└── README.md

---

## 📊 Dataset

This project uses a football dataset prepared using **Roboflow**.  
The dataset includes annotated images for players, referees, and the ball.

> **Note:**  
Datasets and images are **not included** in this repository.  
They can be downloaded separately using Roboflow or a custom download script.

---

## 🧠 Model & Training

The detection model is based on **YOLO**, trained on a custom football dataset.  
Trained model files (`.pt`) are intentionally excluded from the repository to keep it lightweight and reproducible.

---

## ▶️ How to Run

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
