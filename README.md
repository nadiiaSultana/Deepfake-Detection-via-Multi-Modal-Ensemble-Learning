# Deepfake Detection using Ensemble Learning

## Overview
This project implements a lightweight yet effective deepfake detection pipeline using multiple computer vision models and ensemble learning. The system is designed to detect manipulated facial content by leveraging spatial, global, and frequency-domain features.

---

## Dataset
We use a subset of **FaceForensics++**:
- 1000 real images  
- 1000 fake images  
- Preprocessed to 224×224 resolution  

---

## Models Used
- **EfficientNet-B0** → fine-grained spatial artifacts  
- **ResNet50** → hierarchical feature extraction  
- **Vision Transformer (ViT)** → global dependencies  
- **FFT-based CNN** → frequency-domain anomalies  

---

## Method
1. Train models individually  
2. Evaluate using Accuracy, AUC, F1-score  
3. Combine best models using:
   - Weighted averaging  
   - Stacking (meta-learner)

---

## Results

| Model            | Accuracy | AUC  | F1 Score |
|------------------|----------|------|----------|
| EfficientNet-B0  | 0.91     | 0.94 | 0.90     |
| ResNet50         | 0.88     | 0.91 | 0.87     |
| ViT              | 0.90     | 0.93 | 0.89     |
| FFT-CNN          | 0.85     | 0.88 | 0.84     |
| **Ensemble**     | **0.94** | **0.96** | **0.93** |
| **Stacked Model**| **0.95** | **0.97** | **0.94** |

---

## Tech Stack
- Python  
- PyTorch  
- timm  
- OpenCV  
- scikit-learn  
- Google Colab  

---

## Key Insights
- Ensemble learning improves robustness and accuracy  
- CNNs capture spatial artifacts  
- Transformers capture global inconsistencies  
- FFT reveals hidden frequency anomalies  

---

## Future Work
- Train on full dataset  
- Add face detection (MTCNN)  
- Cross-dataset evaluation (Celeb-DF, DFDC)  
- Temporal models for video analysis  

---

## License
For research and educational purposes only.
