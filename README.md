# COMP5405-Digital-Media-Computing
# GitHub README: Dual-Stream Sign Language Translation Framework

This repository contains the implementation of a high-performance, multi-modal framework for **Gloss-Free Continuous Sign Language Recognition (CSLR)**. By integrating structural motion analysis with raw visual context, this system provides a scalable communication bridge for the Deaf and Hard of Hearing (DHH) community.

## Project Overview
Unlike traditional systems that struggle with the fluidity of natural signing, our architecture utilizes a **learned Temporal Boundary Transformer (TBT)** to identify sign transitions end-to-end.This facilitates a seamless mapping from video directly to natural language, bypassing the need for labor-intensive "gloss" annotations.

### Key Features
* **Dual-Stream Extraction**: Combines **RGB Appearance** (via ResNet-50 and Vision Transformer) with **Dense Keypoint Graphs** (via MediaPipe Holistic and ST-GCN).
* **Gloss-Free Decoding**: Maps sign segments directly to text tokens, removing the annotation bottleneck common in academic and industrial pipelines.
* **Learned Segmentation**: Replaces brittle, rule-based "Action Tokenizers" with a data-driven TBT module that handles varying signing speeds and "co-articulation".
* **High-Fidelity Tracking**: Extracts 543 distinct key points, including a 468-point face mesh to capture essential non-manual markers like facial expressions and eye gaze.

---

## System Architecture

1.  **Dual-Stream Feature Extraction**: 
    * **Stream A**: RGB frames are processed through a ResNet-50 backbone and a Visual Transformer.
    * **Stream B**: MediaPipe Holistic extracts skeletal landmarks, which are modeled using a Spatio-Temporal Graph Convolutional Network (ST-GCN) and a BiLSTM.
2.  **Temporal Boundary Transformer (TBT)**: Identifies the start and end of individual signs as a binary sequence labeling task.
3.  **Dual-Stream Feature Fusion (DSFF)**: Uses cross-modal attention to integrate appearance and motion features.
4.  **Gloss-Free Decoder**: A standard 6-layer Transformer decoder generates the final translation.

---

## Comparison with Industry Standards (CSLT-AK)

| Component | CSLT-AK | **Our Proposed System** |
| :--- | :--- | :--- |
| **Sign Segmentation** | Rule-based (Action Tokenizer) | **Learned TBT** |
| **Keypoints** | Basic body pose | **543-pt Holistic** |
| **Feature Streams** | Single stream | **Dual stream (DSFF)** |
| **Gloss Dependency** | Required | **Gloss-free** |
| **Non-manual Features** | Limited | **468-pt face mesh** |

---

## Getting Started

### Implementation Roadmap
*  **Week 7**: Dataset Preparation (PHOENIX-2014T) and Environment Setup.
*  **Week 8**: Development of Feature Extraction Modules (ResNet-50 + ST-GCN).
*  **Week 9**: TBT Module training using pseudo-labels.
*  **Week 10**: Integration of DSFF and Gloss-Free Transformer Decoder.
*  **Week 11**: Full training and evaluation (BLEU, ROUGE, WER).
*  **Week 12**: Finalization and documentation.

---

