# RF Signal Classification using Deep Learning

An AI-based MATLAB project for classifying RF (Radio Frequency) signal images using a fine-tuned MobileNetV2 deep-learning model.

The system accepts an RF signal image / spectrogram-like representation and predicts which of the supported RF signal classes it belongs to.

---

## Project Overview

Radio-frequency signals from different communication systems can have different visual patterns when represented as signal images or spectrograms.

This project uses those visual patterns for classification.

### Basic workflow

```text
RF Signal Image
       ↓
Image Preprocessing
       ↓
224 × 224 × 3 RGB Image
       ↓
MobileNetV2
       ↓
8-Class Classifier
       ↓
Prediction + Confidence
       ↓
Top Predictions
       ↓
UNKNOWN decision when confidence is below threshold
