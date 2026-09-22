# RF Signal Classification using Deep Learning and Grad-CAM

A MATLAB-based deep learning system for classifying Radio Frequency (RF) signal spectrogram images into multiple RF signal categories.

The project uses **MobileNetV2 transfer learning**, RF-specific data augmentation, confidence-based UNKNOWN signal rejection, Grad-CAM explainability, automated evaluation, and a user-friendly visual testing interface.

The system is designed as an end-to-end RF signal classification pipeline rather than only a model-training experiment.

---

# 1. Project Overview

Radio Frequency signals contain important information in the time and frequency domains.

A common way to represent an RF signal is through a **spectrogram**, where:

- X-axis represents time
- Y-axis represents frequency
- Pixel intensity/color represents signal energy

This project treats RF spectrograms as images and uses a deep convolutional neural network to classify them.

The complete system performs:

```text
RF Spectrogram Image
        |
        v
Image Selection
        |
        v
Image Preprocessing
        |
        v
MobileNetV2-based Classifier
        |
        v
Class Probabilities
        |
        v
Confidence Evaluation
        |
        +----------------------+
        |                      |
        v                      v
Confidence >= Threshold    Confidence < Threshold
        |                      |
        v                      v
   KNOWN SIGNAL              UNKNOWN
        |
        v
     Grad-CAM
        |
        v
Visual Result Dashboard
        |
        v
Final Report