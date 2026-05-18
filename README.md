# Dual-Stream Sign Language Recognition (SLR) Framework
### COMP 5405 Digital Media Computing — Group 11 (University of Sydney)
**Contributors:** Yvette Chen (`yche0044`), Javier Marin (`jmar0565`), Anugraha Jawahar (`ajaw0192`)
---

### Master Data-set 
**Kaggle-link:** https://www.kaggle.com/datasets/mrgeislinger/popsign-asl-v1-0-game-train-a-e-signs 
Custom data set was extracted and used for training the model

## 🚀 Project Overview
This repository implements a **Dual-Stream Multi-Modal Architecture** designed to classify isolated American Sign Language (ASL) words from video sequences. 

Traditional single-stream RGB models are highly sensitive to background visual noise, environmental clutter, and illumination shifts. To decouple gestural semantic intent from environmental variance, this framework splits video decoding into two parallel, synchronized pipelines:
1. **Stream A (RGB Video Stream):** Captures high-level visual context, spatial boundaries, and body posture orientations using a pre-trained ResNet-50 backbone fused with a Vision Transformer (ViT) encoder layer.
2. **Stream B (Keypoint Coordinate Stream):** Tracks a dense 543-dimensional coordinate graph per frame (covering face, body pose, and fine-grained finger geometries) extracted via Google MediaPipe Holistic. Motion paths are mapped dynamically over time using a Spatial-Temporal Graph Convolutional Network (ST-GCN) paired with a Bidirectional LSTM.

Features are merged at a mid-level stage via a **Dual-Stream Feature Fusion (DSFF)** multi-head attention module and sequenced through a **Temporal Bottleneck Transformer (TBT)** layer.

---

## 📊 System Limitations & Agile Pivot Analysis

To maintain absolute academic integrity and transparency, this project logs a strict post-mortem on real-world engineering constraints encountered during deployment:

* **The Hardware Wall (Disk Storage Crashes):** Due to the high computational footprint of caching intermediate spatial-temporal video tensors across dense epochs, local environments repeatedly encountered `"Out of Disk Storage"` compiler crashes. To optimize restricted physical compute assets, the team prioritized resources to successfully train and validate the two dominant words—**"Bye"** and **"Can"**.
* **Severe Class Imbalance:** The target subset (`dataset_5_signs`) reflects a massive distribution anomaly. The majority classes make up over 98% of the data volume (`Can`: 537 clips, `Bye`: 463 clips), while minority tokens (`Yes`: 7, `No`: 6, `Help`: 5) suffer from severe data scarcity.
* **The Handedness Bias Variance:** Spatial coordinate mappings suffer from structural mirror-invariance between left-handed and right-handed signers, introducing high model variance during cross-user evaluation.
* **Algorithmic Safety Mechanism (90% Threshold Filter):** Sign language translation demands zero-tolerance for erratic guessing. We engineered a strict **90% Confidence Classification Filter**. If the model's softmax probability drops below this threshold—due to handedness confusion or noise—the runtime rejects the inference token and triggers a clean fallback banner.

---

## 📂 Repository Blueprint

```text
.
├── dataset_5_signs/                    # Isolated word class data directories
│   ├── bye/                            # e.g., ~463 raw .mp4 video clips
│   ├── can/                            # e.g., ~537 raw .mp4 video clips
│   ├── help/
│   ├── no/
│   └── yes/
│
├── Group11_Code.ipynb                 # STAGE 1: Training, tracking, and evaluation
├── sign_language_inference_ui.ipynb   # STAGE 2: Interactive Web UI Deployment
│
├── checkpoints/                        # Serialized .pt model weight storage
└── results/                            # Performance logs, distributions, and confusion matrices

```

---


### Dual-stream TBT architecture and System Workflow
<img width="1024" height="559" alt="image" src="https://github.com/user-attachments/assets/5d275aac-0d89-4ed4-846b-71b94b341f8a" />

## 🛠️ Step-by-Step Setup & Execution Workflow
Execution follows a strict multi-stage lifecycle. **Stage 1** must complete fully to generate the serialized weights asset required by the interface in **Stage 2**.

### 💻 System Prerequisites

Ensure your baseline workspace environment has the following core backends compiled:

```bash
pip install -q opencv-python-headless einops mediapipe ipywidgets tqdm matplotlib pandas scikit-learn torch torchvision

```

### 🔹 Stage 1: Model Training & Evaluation Artifact Generation

**File:** `Group11_Code.ipynb`

Open the training notebook and execute the workspace cells sequentially:

1. **Dependency Verification (Cell 1 & 2):** Compiles framework imports and instantiates directory roots (`./checkpoints`, `./results`). Ensure your raw dataset matches `DATASET_ROOT = Path('./dataset_5_signs')`.
2. **MediaPipe Feature Pre-Extraction (Cell 4):** Automatically pulls asset tasks (`pose_landmarker_full.task` and `hand_landmarker.task`) from Google storage buckets. It extracts raw coordinate structures from your `.mp4` video streams and dumps performance-optimized serialized frames directly into disk-cached directories to prevent loader bottlenecks.
3. **Optimization Loop (Cell 8):** Compiles the multi-stream architecture, partitions files into **Train (75%), Validation (15%), and Test (10%)** sets, and kicks off backpropagation.
4. **Artifact Generation:** Upon identifying optimal validation checkpoints, the loop logs the complete state dictionary array directly to disk path: `./checkpoints/best_10signs.pt`.

---

### 🔹 Stage 2: Deploying the Interactive Video Inference UI

**File:** `sign_language_inference_ui.ipynb`

Once Stage 1 maps out the optimized target model weights, open the interactive deployment notebook:

1. **Point to Model Checkpoint (Step 2):** Verify that the `CHECKPOINT_PATH` explicitly maps to your generated Stage 1 model weights file:
```python
CHECKPOINT_PATH = './checkpoints/best_10signs.pt'
CONFIDENCE_THRESHOLD = 0.90

```


2. **Compile Network Topography (Step 3 & 4):** Run these modules sequentially to mirror the identical PyTorch model layers needed to read and re-hydrate weights from your checkpoint state.
3. **Ignite UI Interface (Step 7):** Execute the final step to launch the front-end application layer right within your Jupyter cell space.

#### 🕹️ UI Operation Flow:

* Click **Upload Video** to select an isolated `.mp4` sign clip from your directory.
* Click **Translate Sign** to run multi-modal inference.
* **Success Route:** If the softmax probability score is strictly $\ge 90\%$, the dashboard prints a green validation alert revealing the matching word token.
* **Fallback Route:** If the score dips under $90\%$, the confidence safety filter halts prediction and logs: `Sorry, I can't translate this word.`

---

## 🔮 Production Roadmap (Future Scope)

1. **Mirror Invariance Normalization:** Introduce a coordinate-flipping pre-processing channel that maps all left-handed coordinate matrices across the Y-axis, normalizing all gesture data to a uniform "right-handed spatial framework."
2. **Cloud Architecture Migration:** Move training loops away from restricted local hardware profiles onto cloud-hosted clusters (AWS EC2 / Google Colab Pro) to bypass physical local disk caching limitations.
3. **Focal Loss Loss-Function Tuning:** Transition from standard Cross-Entropy to Focal Loss to scale up gradients for hard, missing minority items (`Yes`, `No`, `Help`) while smoothing out dominant majority anchors.

