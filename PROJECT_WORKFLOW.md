# Deep Engineering Blueprint & End-to-End System Workflow
### Dual-Stream Sign Language Recognition (SLR) via Temporal Bottleneck Transformers
**COMP 5405: Digital Media Computing — Group 11 (University of Sydney)**

---

This document provides an exhaustive, step-by-step architectural breakdown of the data processing pipelines, network dynamics, optimization schedules, and interactive user interface lifecycles implemented across `Group11_Code.ipynb` and `sign_language_inference_ui.ipynb`. It is intended as a technical manual for our project repository to ensure transparency, reproducibility, and rigorous post-mortem reporting.

---

## 1. Phase I: Spatial-Temporal Data Engineering & Preprocessing

```
[Raw .mp4 Clips] ──► Sub-1KB Skips ──► Uniform Temporal Resampling (T=64)
                             │
            ┌────────────────┴────────────────┐
            ▼                                 ▼
   [RGB Transformation]             [MediaPipe Extraction]
 - 112x112 Resizing               - Pose Landmark (33 pts)
 - Random Horizontal Flip         - Face Landmark (468 pts)
 - Color Jittering                - Left/Right Hands (21 pts x 2)
 - ImageNet Normalization         - Frame Stitching Matrix (543, 3)
            │                                 │
            ▼                                 ▼
 [Dynamic Spatial Tensor]         [GPU Disk Cache (kp_cache_10signs)]
```

### 1.1 Integrity Checks & Video Filtering
1. **Directory Scraping:** The system crawls through the structured root directory (`./dataset_5_signs`), scanning the five target isolated sign word subfolders: `bye`, `can`, `help`, `no`, and `yes`.
2. **Corruption Filtering:** Individual video assets are subjected to binary volume inspection. Any clip dropping beneath a storage allocation threshold of **1KB** is classified as corrupted or empty and skipped automatically to guarantee data loader stream stability.

### 1.2 Multi-Modal Feature Extraction Channels
1. **Temporal Uniform Interpolation:** Given that individual video lengths vary drastically across separate recording takes, clips are resampled uniformly along their time dimension using linear step interpolation to lock down a rigid frame budget of $T = 64$ frames.
2. **Visual Transformations Pipeline (Stream A):**
   * **Inference Layout:** Resized to a uniform $112 	imes 112 	imes 3$ resolution, converted to tensor arrays, and normalized via standardized ImageNet baseline distributions ($\mu = [0.485, 0.456, 0.406]$, $\sigma = [0.229, 0.224, 0.225]$).
   * **Training Layer Regularizations:** To artificially increase boundary entropy and guard against deep spatial memorization, training frames pass through a dynamic augmentation block containing a 50% probability **Random Horizontal Flip** and a **Color Jitter** filter adjusting contrast (0.3), brightness (0.3), and saturation (0.2).
3. **Geometric Keypoint Generation Pipeline (Stream B):**
   * Pre-extraction runs fully offline using Google MediaPipe Holistic tasks.
   * On every discrete frame matrix, the tracker isolates and generates 3D spatial positions ($x, y, z$) for **543 structural nodes**: Pose (33 points), Face Mesh (468 points), Left Hand (21 points), and Right Hand (21 points).
   * Any untracked or occluded hand joints are imputed via frame-to-frame temporal linear interpolation to protect against sudden coordinate drops from injecting gradient anomalies.
   * **Disk Caching Strategy:** To mitigate CPU overhead bottlenecks during real-time training iteration, the pre-computed arrays are dumped as NumPy binaries straight into a persistent disk directory (`./kp_cache_10signs`).

---

## 2. Phase II: Multi-Modal Stream Architecture Topography

The unified network processes parallel geometric matrices and spatial video frames simultaneously using a learned mid-level fusion policy:

```
[Stream A: RGB Video Tensor]                    [Stream B: Keypoint Coordinate Cache]
       │ (B, T, 3, 112, 112)                                │ (B, T, 543, 3)
       ▼                                                    ▼
  ResNet-50 Backbone (Frozen Layers 1-2)             ST-GCN Graph Blocks (Skeleton Convolutions)
       │ (Extracts High-Level Context)                      │ (Extracts Joint Topology Paths)
       ▼                                                    ▼
  Vision Transformer (ViT) Encoder                   Bidirectional LSTM (2-Layer Sequence Model)
       │                                                    │
       ▼                                                    ▼
  Visual Spatial Embeddings (B, T, D)                Geometric Trajectory Embeddings (B, T, D)
       │                                                    │
       └──────────────────────────┬─────────────────────────┘
                                  ▼
                    DSFF (Multi-Head Cross-Attention)
                     - At = Norm(A + Cross(A, B, B))
                     - Bt = Norm(B + Cross(B, A, A))
                                  │
                                  ▼
                    Temporal Bottleneck Transformer (TBT)
                     - Communication via Bottleneck Tokens (B=8)
                     - Reduces Complexity from O(T²) to O(K*T)
                                  │
                                  ▼
                    Linear Classifier & Softmax Block
                     - Enforces 90% Confidence Threshold Constraint
```

### 2.1 Stream A — Visual Spatial Architecture
* **Feature Extraction:** Raw video tensors `[B, T, 3, 112, 112]` are flattened along their temporal dimension into static image batches `[(B * T), 3, 112, 112]` and routed through an ImageNet-weighted **ResNet-50** backbone.
* **Freeze Strategy:** Low-level structural parameters (Layers 1 and 2) are frozen permanently to preserve edge and corner detection primitives. Weights inside Layer 3, Layer 4, and the final linear mapping are unfrozen to tune the network specifically to upper-body gestural context.
* **Sequence Modeling:** Features pass through an embedding projector yielding a dimension scale of $D = 256$, receive 1D learnable positional embeddings to preserve sequence order, and flow into a **Vision Transformer (ViT)** encoder block with multi-head self-attention mechanisms.

### 2.2 Stream B — Geometric Trajectory Architecture
* **Skeletal Graphs Processing:** The persistent keypoint tensor configurations are extracted from disk storage caches and passed through sequential **Spatial-Temporal Graph Convolutional Network (ST-GCN)** structural blocks. Each block pairs regular 1D spatial edge convolutions across connected joint indexes with 1D temporal path convolutions over frames to analyze joint movement patterns.
* **Temporal Modeling:** Graph-pooled joint representations pass directly into a **2-layer Bidirectional Long Short-Term Memory (BiLSTM)** network with a 10% dropout rate. The bidirectional model records the spatial evolution of signs in forward and backward temporal direction simultaneously, projecting the final feature state into an identical coordinate dimension scale of $D = 256$.

### 2.3 Dual-Stream Feature Fusion (DSFF) & Temporal Bottleneck Transformer (TBT)
* Rather than early concatenation or late prediction averaging, the model implements mid-level cross-modal multi-head attention blocks.
* The visual embedding space queries information directly from the geometric tracking matrix, and vice versa:
  $$A_t = 	ext{LayerNorm}(A + 	ext{MultiHeadAttention}(A, B, B))$$
  $$B_t = 	ext{LayerNorm}(B + 	ext{MultiHeadAttention}(B, A, A))$$
* The combined representations are concatenated and projected down to dimension size $D = 256$, then channeled into the **Temporal Bottleneck Transformer (TBT)**. 
* To prevent quadratic scaling computational footprints ($O(T^2)$), cross-attention operations are forced to interact through a small, dense sequence of **8 learnable bottleneck tokens**. This lowers the sequence model's computation overhead to linear complexity ($O(K \cdot T)$), making the network suitable for real-time edge processing.

---

## 3. Phase III: Strategic Optimization & System Post-Mortem

### 3.1 Data Splitting Policy
The global asset index maps are randomly segregated into strict partitions to evaluate generalization capability cleanly:
* **Train Set (75%):** Used for gradient updates and weight updates.
* **Validation Set (15%):** Tracks hyperparameter validation and selects the top performing checkpoint epochs.
* **Test Set (10%):** Locked away until final post-training evaluation to record unseen execution accuracy.

### 3.2 Optimization Space Configuration
* **Optimization Framework:** AdamW optimizer with a strict base learning rate of $3 	imes 10^{-4}$ and an L2 regularization weight decay configuration of $1 	imes 10^{-4}$.
* **Objective Function:** Cross-Entropy loss modified with a **Label Smoothing factor of 0.1** to artificially soften hard target truth vectors, guarding against overconfident boundary predictions.
* **Learning Rate Schedule:** Implements a linear warmup window over the first 50 global optimization steps, followed by a **Cosine Annealing decay pattern** to guide convergence precisely toward deep local minima.

### 3.3 Hardware Post-Mortem & Agile Constraints
During system deployment, the compiler environment encountered repetitive **"Out of Disk Storage"** terminal crashes. Storing and processing massive, multi-modal spatial-temporal video tensors across dense epochs completely exhausted local disk caching space. 

To maintain system integrity and preserve limited local compute resources, the team executed an agile development pivot: training vectors were targeted to resolve the two dominant words—**"Bye"** and **"Can"**—which represented 98.23% of the overall sample repository volume. The model successfully converged on these majority targets (achieving **90.8% peak validation accuracy** at epoch 8), while minority tokens with fewer than 10 samples total ("Help", "No", "Yes") were zeroed out due to severe data scarcity.

---

## 4. Phase IV: Interactive UI & Inference Lifecycle

```
[User .mp4 Video Clip Upload] ──► Read Binary Stream ──► Load Saved Model State (best_10signs.pt)
                                                                 │
                                                                 ▼
                                                  Parallel Feature Extraction
                                                   - Stream A: Frame Resize & Normalization
                                                   - Stream B: Live MediaPipe Graph Mapping
                                                                 │
                                                                 ▼
                                                  Unified Multi-Stream Inference
                                                                 │
                                                                 ▼
                                                    Softmax Probability Check
                                                                 │
                                    ┌────────────────────────────┴────────────────────────────┐
                                    ▼ Probability >= 90%                                      ▼ Probability < 90%
                        [Render Green Success Screen]                             [Render Red Fallback Card]
                        - Output: Upper-case Word Token                           - Output: "Sorry, I can't translate..."
                        - Render Confidence Metric Bar                            - Log Top Prediction Failure Info
```

When deployed via `sign_language_inference_ui.ipynb`, the inference script instantiates an interactive application loop operating via the following runtime stages:

1. **User Upload Interfacing:** The dashboard sets up a front-end `FileUpload` notebook widget filtering specifically for `.mp4`, `.avi`, or `.mov` assets.
2. **Re-hydrating State Checkpoints:** The system opens the serialized training output (`./checkpoints/best_10signs.pt`), reconstructs the dual-stream transformer layout from saved model definitions, and maps the loaded parameter weights onto the runtime CPU or CUDA core arrays.
3. **Inference Loop Processing:**
   * Unpacks the uploaded video file stream into a temporary directory path.
   * Extracts matching temporal frames and routes them through parallel visual transformations and live MediaPipe keypoint trackers.
   * Passes the compiled input matrices into the dual-stream transformer to execute a single inference pass.
4. **Enforcing the Confidence Threshold Constraint:**
   * The terminal maps logits output through a `Softmax` layer to determine class probability strings.
   * **The Success Route ($\ge 90\%$):** If the highest prediction score clears our **90% Confidence Threshold Safety Filter**, the UI updates dynamically with a prominent green confirmation alert showing the validated translation token and a confidence bar chart.
   * **The Fallback Route ($< 90\%$):** If tracking anomalies, motion blur, or user handedness variations cause the prediction confidence to drop even slightly to 89%, the safety filter rejects the translation completely. The UI renders a red alert block stating: `Sorry, I can't translate this word.`, preventing the application from outputting inaccurate guesses.