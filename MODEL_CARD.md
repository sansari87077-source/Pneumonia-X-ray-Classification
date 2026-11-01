# Model Card — Pneumonia Detection on Chest X-Ray Images Using Deep Learning

## Model Overview
- **Project Ref:** Ref X-Ray Exp / DL-005
- **Authors:** Eda AYDIN, Nesibe Betül DÖNER, Okan TOPAL, Berkay DİRİL
- **Architecture:** Custom 5-Block Deep Convolutional Neural Network (Conv2D, BatchNormalization, MaxPooling2D, Dropout, Dense)
- **Framework:** TensorFlow / Keras
- **Input Specifications:** $150 \times 150 \times 1$ Grayscale image, normalized to $[0, 1]$
- **Output:** Single node sigmoid probability ($0 = \text{NORMAL}$, $1 = \text{PNEUMONIA}$)
- **Status:** Educational demonstration prototype

## Training Parameters (Slide 24)
- **Optimizer:** Adam (Initial Learning Rate = 0.001)
- **Scheduler:** `ReduceLROnPlateau(monitor='val_accuracy', factor=0.3, patience=2, min_lr=1e-5)`
  - Epoch 1–4: $lr = 0.0010$
  - Epoch 5–12: $lr = 0.0003$
  - Epoch 13–14: $lr = 0.00009$
  - Epoch 15: $lr = 0.000027$
- **Total Epochs:** 15
- **Augmentation Pipeline:** Rotation ($\pm 30^\circ$), Zoom ($20\%$), Height/Width Shifts ($10\%$), Horizontal Flip.

## Benchmark Performance (Slides 25, 28, 29)
- **Final Model Loss:** 0.1278
- **Final Model Accuracy:** 95.25% (5,578 / 5,856 correct)
- **Confusion Matrix:**
  - True Negative (TN): 1,531
  - False Positive (FP): 52
  - False Negative (FN): 226
  - True Positive (TP): 4,047
- **Class Metrics:**
  - Pneumonia (Class 1): Precision: 0.99 | Recall: 0.95 | F1-score: 0.97
  - Normal (Class 0): Precision: 0.87 | Recall: 0.97 | F1-score: 0.92

## Intended Use & Limitations
- **Intended Use:** Teaching computer vision methodologies, preprocessing grayscale radiographs, handling class imbalance ($74.3\%$ vs $25.7\%$), and inspecting false-negative diagnostic risks.
- **Out-of-Scope:** Clinical diagnostic decision making, emergency triage, or hospital intake decisions.