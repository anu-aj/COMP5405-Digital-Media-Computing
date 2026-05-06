# Real-Time ASL Translator using Dual-Stream SLR Model

## Project Overview

This project implements a **real-time American Sign Language (ASL) recognition system** using a **Dual-Stream Sign Language Recognition (SLR) model** trained on the WLASL100 dataset.

The system combines:

* **RGB video frames** (visual appearance)
* **MediaPipe keypoints** (hand/body landmark movement)

The final output is a live ASL prediction prototype that works inside **Google Colab** using webcam input.

---

# Project Workflow

## Step 1 — Train the Model

First, run the original training notebook:

```text
wlasl100_dual_stream_slr.ipynb
```

This notebook:

* Loads and preprocesses the WLASL100 dataset
* Extracts MediaPipe keypoints
* Trains the dual-stream SLR model
* Evaluates performance
* Saves the trained model checkpoint

---

## Step 2 — Save the Trained Model

After training completes, ensure the following checkpoint file is created:

```text
checkpoints/best_wlasl100.pt
```

This file contains the trained model weights and will be used for live inference.

---

# Colab-Based Real-Time ASL Translator

## Why Colab Instead of Tkinter?

Tkinter GUIs and direct webcam access using:

```python
cv2.VideoCapture(0)
```

do not work reliably inside Google Colab.

Instead, the project uses:

* Browser webcam capture
* OpenCV frame processing
* MediaPipe keypoint extraction
* Notebook-based live prediction display

This makes the project fully compatible with Google Colab.

---

# Step 3 — Open the Colab-Friendly Notebook

Use the provided notebook:

```text
wlasl100_colab_asl_translator.ipynb
```

This notebook performs:

* Webcam capture from browser
* Frame preprocessing
* Keypoint extraction
* Sliding-window buffering
* Real-time ASL prediction
* Prediction display inside notebook output

---

# Step 4 — Upload Required Files to Colab

Upload the following files into your Colab session:

## Required Files

```text
checkpoints/best_wlasl100.pt
```

If your class mapping is stored separately, also upload:

```text
IDX2CLASS mapping file
```

---

# Step 5 — Install Dependencies

Run the dependency installation cells first.

Required libraries include:

```python
opencv-python
mediapipe
torch
torchvision
numpy
Pillow
```

---

# Step 6 — Run the Notebook Cells Sequentially

Execute the notebook in the following order:

1. Install dependencies
2. Import libraries
3. Load model checkpoint
4. Initialize MediaPipe extractor
5. Start webcam capture
6. Begin live inference

---

# Real-Time Inference Pipeline

The system follows this processing pipeline:

```text
Webcam Frame
      ↓
MediaPipe Keypoint Extraction
      ↓
RGB Frame Preprocessing
      ↓
64-Frame Sliding Window Buffer
      ↓
Dual-Stream Model Inference
      ↓
Predicted ASL Word
```

---

# Important Technical Notes

## 1. Input Shape Consistency

The preprocessing inside the Colab notebook must exactly match the training notebook.

### RGB Stream Shape

```text
(1, 64, 3, 112, 112)
```

### Keypoint Stream Shape

```text
(1, 64, 543, 3)
```

---

## 2. Frame Resize

Frames must be resized to:

```python
(112, 112)
```

before inference.

---

## 3. Sliding Window Logic

The model predicts after collecting:

```text
64 frames
```

Once prediction is complete:

* The oldest frame is removed
* The newest frame is added
* Prediction continues continuously

---

# Testing the Model

Test the system using:

* Clear hand gestures
* Stable lighting
* Plain background
* Front-facing webcam position

Start with a few known WLASL100 signs to validate predictions.

---

# Recommended Demo Scope

If prediction quality is inconsistent, present the project as:

> A prototype real-time ASL recognition system using a dual-stream deep learning architecture.

Focus on:

* System architecture
* Real-time inference pipeline
* Integration of RGB + keypoint streams
* Human-computer interaction aspects

---

# Known Limitations

Current limitations include:

* Limited vocabulary (WLASL100 subset)
* Sensitivity to lighting conditions
* Webcam angle dependency
* Colab webcam latency
* Temporal prediction instability for fast gestures

---

# Future Improvements

Possible future enhancements:

* Streamlit-based web application
* Desktop GUI using Tkinter or PyQt
* Larger ASL vocabulary
* Sentence-level recognition
* Transformer-based temporal models
* Real-time smoothing and confidence filtering
* Cloud deployment

---

# Suggested GitHub Repository Structure

```text
project-root/
│
├── notebooks/
│   ├── wlasl100_dual_stream_slr.ipynb
│   └── wlasl100_colab_asl_translator.ipynb
│
├── checkpoints/
│   └── best_wlasl100.pt
│
├── README.md
├── requirements.txt
└── assets/
```

---

# Suggested README Sections

Your GitHub README should include:

* Project Overview
* Dataset Information
* Model Architecture
* Real-Time Inference Pipeline
* Installation Steps
* Usage Instructions
* Demo Screenshots/GIFs
* Limitations
* Future Work

---

# Technologies Used

* Python
* PyTorch
* OpenCV
* MediaPipe
* NumPy
* Google Colab
* Deep Learning
* Computer Vision
* Human-Computer Interaction

---

# Final Submission Recommendation

For the best presentation:

## Include:

* Training notebook
* Colab demo notebook
* Saved checkpoint
* Screenshots/GIFs of predictions
* Architecture diagram
* README documentation

## Demo Flow:

1. Explain the dual-stream architecture
2. Show webcam capture
3. Demonstrate prediction pipeline
4. Show live ASL recognition
5. Discuss limitations and future improvements

---

# Conclusion

This project demonstrates the integration of:

* Deep learning
* Computer vision
* MediaPipe landmark extraction
* Real-time inference systems
* Human-computer interaction concepts

to build a prototype ASL translation system capable of recognizing sign language gestures in real time.
