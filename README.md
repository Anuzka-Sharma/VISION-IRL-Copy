<<<<<<< HEAD
<<<<<<< HEAD
# VISION IRL
=======
=======
>>>>>>> 648f8905b368c0dfda6a4e4ac8b3d074198c8b36
# VISION-IRL-Copy

````markdown
# 👁️‍🗨️ VISION IRL – Offline Assistive AI for the Visually Impaired

**VISION IRL** is an **offline-first, privacy-focused assistive system** built to help visually impaired users interact with their surroundings through **AI-driven perception**. It runs entirely **locally** on Snapdragon® X Elite hardware, making use of **Qualcomm’s NPU acceleration** through the **Snapdragon® AI Stack** and **QNN (Qualcomm® Neural Network)** tools.

---

## 🚀 Key Features

- 🎯 Real-time Object Detection with **YOLOv8**
- 🧍 Pose Detection via **MediaPipe**
- 👁️ Depth Perception using **Intel RealSense™**
- 🔊 Fully Offline Speech I/O using **Vosk + pyttsx3**
- 🧠 Face & Voice-based User Authentication
- ⚙️ Runs fully on-device — **no internet required**
- ⚡ Optimized for **Snapdragon X Elite** with NPU acceleration via **QNN + AI Hub**

---

## 💻 System Requirements

| Component          | Requirement                         |
|--------------------|-------------------------------------|
| **Device**          | Snapdragon® X Elite (e.g. Dell 7455)|
| **OS**              | Windows 11 ARM64                   |
| **Python**          | 3.12.x                             |
| **Camera**          | Intel RealSense D435i (or similar) |
| **NPU Runtime**     | Snapdragon® AI SDK + QNN           |

---

## 📦 Installation

### 1. Clone the Project
```bash
git clone https://github.com/Anuzka-Sharma/VISION-IRL-Copy.git
cd VISION-IRL-Copy
````

### 2. Create and Activate Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

If `PyAudio` fails:

```bash
pip install pipwin
pipwin install pyaudio
```

---

## 🔌 Using Snapdragon® NPU (QNN) Acceleration

### ✅ Requirements

* [Snapdragon® AI SDK](https://developer.qualcomm.com/software/ai-stack)
* [Qualcomm® AI Hub account](https://aihub.qualcomm.com)
* Compatible `.onnx` or `.tflite` models (you can convert YOLO, Whisper, etc.)
* Optional: [AnythingLLM + NPU Edge Chat App](https://github.com/Qualcomm-AI-research)

---

### 🛠️ Steps to Run with QNN Acceleration

1. **Install Snapdragon AI SDK for Windows**

   * Download from: [https://developer.qualcomm.com/software/ai-stack](https://developer.qualcomm.com/software/ai-stack)
   * Follow the SDK documentation to set up the QNN runtime.

2. **Convert Model for QNN**

   * Use AI Hub or QNN Tools to convert models to `.qnn` format:

   ```bash
   qnn-convert --onnx your_model.onnx --output-dir ./models/qnn
   ```

3. **Enable QNN Backend in Code**

   * In `object_detection.py` or `speech_model.py`:

   ```python
   from qti.ai.model.api import QNNEngine
   model = QNNEngine("models/qnn/yolov8.qnn", backend="npu")
   ```

4. **Verify Acceleration**

   * Use `qnn-inspect` or runtime logs to confirm that the NPU is running ops (e.g. `Compute Unit(s): npu (851 ops)`).

---

## 🧪 Real-Time Capabilities

| Function         | Tech Used         | Accelerated                        |
| ---------------- | ----------------- | ---------------------------------- |
| Object Detection | YOLOv8 (ONNX/QNN) | ✅ (via QNN)                        |
| Speech-to-Text   | Whisper / Vosk    | ✅ (Whisper on QNN or CPU fallback) |
| Face Auth        | dlib / FaceNet    | ❌ (CPU-only)                       |
| Depth Estimation | RealSense SDK     | ❌ (native pipeline)                |
| Pose Detection   | MediaPipe         | ✅ (GPU/CPU)                        |

---

## 🧠 Offline AI Pipeline

```text
[Camera Input]
     ↓
[YOLOv8 on NPU] → Object Labels
     ↓
[RealSense Depth Map] → Distance
     ↓
[Pose Detection (MediaPipe)]
     ↓
[Speech Recognition (Vosk)] ← Microphone
     ↓
[Command Parser + TTS (pyttsx3)] → Speaker
```

---

## 🧑‍💻 Project Structure

```
VISION-IRL/
├── backend/
│   ├── object_detection.py
│   ├── speech_model.py
│   ├── pose_module.py
│   └── auth/
├── models/
│   ├── yolov8.onnx
│   └── yolov8.qnn
├── assets/
├── main.py
├── requirements.txt
└── README.md
```

---

## 🧰 Toolchain Overview

| Tool                               | Purpose                    |
| ---------------------------------- | -------------------------- |
| `ultralytics`                      | YOLOv8 object detection    |
| `pyrealsense2`                     | RealSense D435i for depth  |
| `vosk`                             | Offline STT                |
| `pyttsx3`                          | Offline TTS                |
| `kivy`                             | GUI Framework              |
| `qnn` / `QNN SDK`                  | NPU inference acceleration |
| `SpeechRecognition`                | Optional fallback STT      |
| `torch`, `onnx`, `huggingface_hub` | Model development          |

---

## 🧠 LLM Chatbot Option (NPU)

You can also integrate a **local LLM chatbot** using:

* ✅ [Snapdragon NPU Edge Chat App](https://github.com/Qualcomm-AI-research)
* ✅ `AnythingLLM` with QNN-accelerated models
* Use this for **natural voice Q\&A**, **scene descriptions**, and **contextual help**

---

## 📜 License

MIT License – free for non-commercial use with attribution.

---

## 🤝 Acknowledgements

* Qualcomm® AI Stack + AI Hub
* Intel® RealSense SDK
* Ultralytics YOLOv8
* Whisper / Vosk
* MediaPipe by Google
* Kivy Open Source Framework

---

**Built for a world that everyone can experience. Offline. Real. Empowering. — VISION IRL**

```

---

Let me know if you'd like to include:
- A link to a sample `.onnx` → `.qnn` conversion script
- Example logs of NPU-accelerated output
- A badge-style intro for GitHub, or
- A minimal version for submission to judges.
```
<<<<<<< HEAD
>>>>>>> 648f8905b368c0dfda6a4e4ac8b3d074198c8b36
=======
>>>>>>> 648f8905b368c0dfda6a4e4ac8b3d074198c8b36
