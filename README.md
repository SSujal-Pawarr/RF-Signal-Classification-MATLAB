# RF Signal Classification using Deep Learning and Grad-CAM

A deep learning-based RF signal classification system developed in MATLAB using transfer learning with MobileNetV2.

The system classifies RF spectrogram images into predefined signal classes, provides confidence scores, detects potentially unknown/unseen signals using a calibrated confidence threshold, and generates Grad-CAM visual explanations for accepted known signals.

---

## 1. Project Overview

Radio Frequency (RF) signals can be represented as spectrogram images containing time-frequency information.

This project uses a convolutional neural network with transfer learning to classify RF spectrogram images into eight predefined RF signal categories.

The system was designed as an end-to-end pipeline rather than only a model-training experiment.

The complete pipeline is:

User selects image
        ↓
Image preprocessing
        ↓
MobileNetV2-based RF classifier
        ↓
Class probabilities
        ↓
Confidence evaluation
        ↓
UNKNOWN rejection
        ↓
Final system decision
        ↓
Grad-CAM explanation for known signals
        ↓
Visual result window
        ↓
Final text report


---

## 2. Objectives

The main objectives of the project are:

- Classify RF signal spectrogram images.
- Use transfer learning to improve classification performance.
- Support multiple RF signal categories.
- Provide confidence scores for predictions.
- Detect potentially unknown/unseen signals.
- Avoid generating misleading Grad-CAM explanations for rejected UNKNOWN signals.
- Provide visual explanations using Grad-CAM.
- Save classification results and explanations.
- Provide a user-friendly image-selection and result interface.
- Build a modular and reusable MATLAB implementation.


---

## 3. RF Signal Classes

The trained model contains eight RF signal classes:

1. FM
2. Bluetooth
3. WiFi
4. Cellular
5. LoRa
6. AIS
7. Airband
8. RS41-Radiosonde

The class order used internally by the trained MATLAB network is:

1. RS41-Radiosonde
2. airband
3. ais
4. bluetooth
5. cellular
6. fm
7. lora
8. wifi


---

## 4. Dataset

The project contains training and validation RF spectrogram images.

### Training Dataset

| Class | Images |
|---|---:|
| RS41-Radiosonde | 193 |
| airband | 46 |
| ais | 79 |
| bluetooth | 147 |
| cellular | 282 |
| fm | 515 |
| lora | 417 |
| wifi | 205 |
| **Total** | **1,884** |

### Validation Dataset

| Class | Images |
|---|---:|
| RS41-Radiosonde | 41 |
| airband | 9 |
| ais | 16 |
| bluetooth | 31 |
| cellular | 60 |
| fm | 110 |
| lora | 89 |
| wifi | 43 |
| **Total** | **399** |


---

## 5. Model Architecture

The project uses:

**MobileNetV2 with transfer learning**

The pretrained MobileNetV2 network was adapted for the RF signal classification task.

### Input

```text
224 × 224 × 3