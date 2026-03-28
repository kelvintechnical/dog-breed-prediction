# Dog Breed Prediction (CNN + Streamlit)

![Python](https://img.shields.io/badge/Python-3.13-blue?style=flat-square&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=flat-square&logo=tensorflow)
![Keras](https://img.shields.io/badge/Keras-2.x-red?style=flat-square&logo=keras)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-ff4b4b?style=flat-square&logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

An image classification project that predicts dog breeds using a custom convolutional neural network (CNN) trained from scratch, then served through a Streamlit web app.

This project intentionally avoids transfer learning to build deeper intuition around:
- convolutional feature extraction
- architecture design tradeoffs
- regularization on small datasets
- full training-to-deployment workflow

> Live demo URL: add your Streamlit Cloud link here when deployed.

---

## Project Pipeline

![Pipeline](https://github.com/user-attachments/assets/c74a9be2-96c7-4532-99da-a624c24910f4)

---

## Results Snapshot

| Metric | Value |
|---|---|
| Classes | 3 (Scottish Deerhound, Maltese, Bernese Mountain Dog) |
| Epochs | 20 |
| Optimizer | Adam (`lr = 0.0001`) |
| Loss | Categorical Crossentropy |
| Train / Val / Test Split | 72% / 18% / 10% |

Add these when available:
- Validation accuracy
- Test accuracy
- Training curves
- Confusion matrix

Recommended assets:
- `assets/training_curves.png`
- `assets/confusion_matrix.png`

---

## Demo

Workflow:
1. Upload a dog image
2. Click **Predict**
3. View predicted breed and confidence score

Optional: add a demo GIF or screenshot at `assets/demo.gif`.

---

## Architecture Rationale

The model is built from scratch (instead of using ResNet, MobileNet, or EfficientNet) to emphasize learning fundamentals over benchmark optimization.

### Key Design Choices

- **Decreasing filter counts (`64 -> 32 -> 16 -> 8`)**  
  Compresses representations progressively and limits model capacity.
- **Mixed kernel sizes (`5x5`, `3x3`, `7x7`, `5x5`)**  
  Captures both local texture details and broader structural patterns.
- **L2 regularization on convolutional and dense layers**  
  Reduces overfitting risk on a small, limited-class dataset.
- **Low learning rate (`0.0001`)**  
  Encourages stable updates in a noisy, small-data optimization setting.

```text
Input (224x224x3)
  -> Conv2D(64, 5x5, relu) + MaxPool2D
  -> Conv2D(32, 3x3, relu, L2) + MaxPool2D
  -> Conv2D(16, 7x7, relu, L2) + MaxPool2D
  -> Conv2D(8,  5x5, relu, L2) + MaxPool2D
  -> Flatten
  -> Dense(128, relu, L2)
  -> Dense(64,  relu, L2)
  -> Dense(3, softmax)
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.13 |
| Deep Learning | TensorFlow / Keras |
| Model Type | Convolutional Neural Network (CNN) |
| Web App | Streamlit |
| Data Source | [Kaggle: Dog Breed Identification](https://www.kaggle.com/c/dog-breed-identification) |
| Training Environment | Google Colab |
| Serving Environment | Local (Streamlit) |

---

## Project Structure

```text
dog_breed_app/
├── Dog_Breed_Prediction.ipynb   # Full training pipeline (Colab)
├── main_app.py                  # Streamlit web application
├── requirements.txt             # Project dependencies
├── dog_breed_pipeline.svg       # Architecture diagram
├── assets/                      # Training curves, confusion matrix, demo GIF
└── README.md
```

`dog_breed_model.keras` is excluded in `.gitignore` due to file size.  
Run `Dog_Breed_Prediction.ipynb` in Colab to regenerate it.

---

## Getting Started

### 1) Clone the repository

```bash
git clone https://github.com/kaboroinformatics/dog-breed-prediction.git
cd dog-breed-prediction
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

### 3) Train and export the model

Open `Dog_Breed_Prediction.ipynb` in [Google Colab](https://colab.research.google.com/), run all cells, and place the generated `dog_breed_model.keras` file in the project root.

### 4) Run the Streamlit app

```bash
python -m streamlit run main_app.py
```

Then open [http://localhost:8501](http://localhost:8501).

---

## Lessons Learned

- **Overfitting appears quickly on small datasets.** L2 regularization and conservative learning rates helped more than deeper architectural complexity.
- **Preprocessing quality strongly affects training behavior.** Consistent resize to `224x224` and normalization to `[0, 1]` improved convergence stability.
- **Building from scratch improves model intuition.** Manual choices around filters, pooling, and capacity made tradeoffs much clearer than using pretrained backbones.

---

## Future Work

- [ ] Scale from 3 breeds to 120 breeds with transfer learning (MobileNetV2 or EfficientNet-B0)
- [ ] Add data augmentation (rotation, flipping, color jitter)
- [ ] Deploy to Streamlit Community Cloud
- [ ] Add Docker support for reproducible deployment
- [ ] Add Grad-CAM visualizations for model interpretability
- [ ] Benchmark custom CNN vs pretrained baselines

---

## Author

**Kelvin R. Tobias**  
Software engineer transitioning into AI/ML, with a research focus on computational biology and latent-based directed evolution.

- B.S. Software Engineering, Western Governors University (2026)
- M.S. AI Engineering Candidate, WGU (target: Dec 2026)

[LinkedIn](https://www.linkedin.com/in/kelvin-r-tobias-211949219/) | [GitHub](https://github.com/kelvintechnical) | [Kelvinintech Consulting LLC](https://kelvinintech.com)

---

## License

[MIT License](LICENSE) - free to use, modify, and distribute with attribution.
