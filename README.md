# self_supervised_learning_rotation_angle_prediction_project
# Self-Supervised Learning: Rotation Prediction on MNIST

## 🧠 Core Idea
- Train a CNN to predict image rotation angle (0°, 90°, 180°, 270°).
- No human labels needed → labels are generated automatically.
- The CNN learns general features useful for real classification.

## 📂 Project Steps
1. `train_ssl.py`: Train self-supervised model (pretext task).
2. `evaluate_downstream.py`: Freeze features → train simple classifier on few labels.
3. Compare downstream accuracy to raw baseline.

## 🚀 Run
```bash
python train_ssl.py
python evaluate_downstream.py
