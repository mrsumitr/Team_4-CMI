# BFRB Detection with Helios Wrist Sensor
**Affiliation:** Child Mind Institute dataset / Research Project  
**Year:** 2025

## Project Overview
This project develops predictive models to detect Body-Focused Repetitive Behaviors (BFRBs) such as hair pulling, skin picking, and nail biting using data from the **Helios** wrist-worn device. Helios includes an IMU plus additional sensors (5 thermopiles, 5 time-of-flight proximity sensors). The competition evaluates models on two conditions: (1) IMU-only data and (2) full-sensor data (IMU + thermopiles + ToF).

## Problem Statement
Build models that:
1. Classify whether a recorded gesture is **BFRB-like (target)** or **non-BFRB-like (non-target)**.  
2. Predict the **specific BFRB gesture** (multi-class), collapsing all non-target gestures into a single `non_target` class.

Evaluation metric: average of binary F1 (target vs non-target) and macro F1 across gesture classes.

## Dataset
- Wrist-worn Helios recordings with IMU, 5 thermopiles, and 5 time-of-flight sensors.
- Each participant performed 18 gestures (8 BFRB-like, 10 non-BFRB-like) across up to 4 body positions (sitting, sitting leaning forward, lying on back, lying on side).
- Each gesture sequence contains three phases: **Transition → Pause → Gesture**.

## Sensors
- IMU: accelerometer + gyroscope (standard motion features)
- Thermopiles (5): detect body heat / thermal signatures
- Time-of-Flight (5): proximity / distance to body parts

## Approach (Suggested / Implemented)
1. **Preprocessing**
   - Synchronize sensor streams, trim to gesture window, normalize per-sensor.
   - Compute time- and frequency-domain features per sensor (mean, std, peak, energy, spectral bands) or use raw windows for deep learning.
2. **Modeling**
   - Baselines: Random Forest / XGBoost on engineered features.
   - Deep models: 1D-CNN, LSTM, or Transformer on multivariate time series.
   - Two-model evaluation strategy:
     - Model A: IMU-only input (for IMU-only test partition).
     - Model B: IMU + thermopiles + ToF (for full-sensor partition).
   - Optionally: multimodal fusion, sensor-dropout augmentation, and class-balanced sampling.
3. **Training & Validation**
   - Use participant-wise splits (leave-subjects-out) to test generalization.
   - Evaluate both binary F1 and macro F1; monitor per-class F1 to detect collapse into `non_target`.

## Results (Summary)
Validation Accuracy: 0.6343558282208589
Validation Macro F1: 0.645763211839946
## How to run
1. Create a Python environment:
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Repository Structure (recommended)
```
.
├── LICENSE
├── README.md
├── cmi-1.ipynb
└── presentation.pptx
```

## Notes & Tips
- Make sure to collapse all non-target gestures into `non_target` for the gesture-class evaluation.
- For fair comparison of IMU vs full sensors, train/evaluate separate models or use an input-mask mechanism.
- Report per-class confusion matrix and per-sensor ablation studies to quantify the value of thermopiles and ToF.

## References
- Garey, J. (2025). *What Is Excoriation, or Skin-Picking?* Child Mind Institute.
- Martinelli, K. (2025). *What is Trichotillomania?* Child Mind Institute.

---