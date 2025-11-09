# 🚀 AI-Powered Django Web App

**Created by: Shahzad Abdul Majeed**

## 📌 Project Overview
This project is a **Django-based AI-powered web application** that integrates multiple deep learning models to perform **real-time object detection, face recognition, image classification, and math equation solving**. It showcases the power of AI in real-world applications with advanced accuracy and efficiency.

🌟 **And there's more!** I'm actively working on exciting **AI solutions**, including **AI video editing, AI voice generation, and many more innovative features!** Stay tuned! 🔥

---

## 🔥 Key Features & How They Work

### 🟢 Live Object Detection (YOLOv8)
- **How It Works:**
  - Uses the **YOLOv8 model**, a state-of-the-art object detection algorithm.
  - Captures video frames in real time using OpenCV.
  - Processes each frame with YOLOv8 to detect multiple objects simultaneously.
  - Overlays **bounding boxes and class labels** on detected objects.
  - Displays the processed video feed on the web interface.

![Screenshot 2025-03-15 212458](https://github.com/user-attachments/assets/81814c6b-6c8b-478a-8103-91469fec2536)


---

### 🟢 Face Recognition (DeepFace)
- **How It Works:**
  - Uses the **DeepFace model** to analyze human faces.
  - Detects faces and extracts features such as **age, gender, and emotions**.
  - Outputs the detected attributes in real-time for images and webcam streams.
  - Can be extended to recognize specific individuals with training.

![stock-2](https://github.com/user-attachments/assets/4d414342-4325-4e6a-bfeb-b8e4928b3e0a)


---

### 🟢 Image Classification (Faster R-CNN Inception ResNet V2)
- **How It Works:**
  - Uses the **Faster R-CNN Inception ResNet V2 model**, a highly accurate image classification network.
  - Analyzes input images and identifies objects with confidence scores.
  - Returns classification labels with precise bounding boxes.
  - Can be used for object tracking and anomaly detection.

![Screenshot 2025-03-15 213038](https://github.com/user-attachments/assets/ee308b1c-15ce-4b73-83d1-5861b6e7930d)


---

### 🟢 Math Solver (Qwen2.5-0.5B)
- **How It Works:**
  - Uses **OCR (Optical Character Recognition)** to extract math equations from images.
  - Converts extracted text into **LaTeX format** for structured representation.
  - Passes the LaTeX expression to the **Qwen2.5-0.5B model** for solving.
  - Provides **accurate step-by-step solutions** for algebra, calculus, and more.
  - Outputs results in a user-friendly format on the web app.



---

## ⚙️ Tech Stack
- **Frontend:** Django (HTML, CSS, JavaScript)
- **Backend:** Django + FastAPI for real-time AI model inference
- **Deep Learning Models:** YOLOv8, DeepFace, Faster R-CNN, Qwen2.5-0.5B
- **Libraries:** OpenCV, TensorFlow, PyTorch, FiftyOne, NumPy, Pandas, SymPy
- **Math Processing:** Pix2Text, DeepSeek Math, SymPy

---

## 🛠️ Setup Instructions
1. Clone the repository:
   ```bash
   git clone https://github.com/MShahzadAbdulmajeed/AI_Django_Website.git
   cd AI_Django_Website
   ```
2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. Run the Django server:
   ```bash
   python manage.py runserver
   ```

---

## ❌ Why Isn't This Project Live?
💡 **AI models require significant computational resources!**

Hosting deep learning models like YOLOv8, DeepFace, and Faster R-CNN requires **high-performance GPUs**, which are expensive to run on cloud servers. Unfortunately, free hosting options **do not support such heavy models**, and paid servers are **too costly** to maintain for a personal project.

However, you can **run it locally** by following the setup instructions above! 🚀

---

## 📞 Contact Me for AI Projects & Collaborations!
I’m open to exciting AI collaborations and freelance projects! Let’s build the future together. 🚀

📧 **Email:** shahzad.abdulmajeed4894@gmail.com  
📱 **Phone:** +92 320 6236425  
🔗 [Linkedin](www.linkedin.com/in/shahzad-abdulmajeed-618887220)

---

## 💬 Let's Connect!
If you find this project interesting, feel free to ⭐ star this repository and contribute! 😊

---

## 📜 License
This project is **open-source** under the MIT License. Feel free to explore, modify, and enhance it!

