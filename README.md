Dog Breed Prediction — CNN + Streamlit
![Python](https://img.shields.io/badge/Python-3.13-blue?style=flat-square&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=flat-square&logo=tensorflow)
![Keras](https://img.shields.io/badge/Keras-2.x-red?style=flat-square&logo=keras)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-ff4b4b?style=flat-square&logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
A from-scratch convolutional neural network that classifies dog breeds from images, deployed through an interactive Streamlit web app. Built intentionally without transfer learning to develop a ground-up understanding of CNN feature extraction, regularization, and end-to-end ML pipeline design.
<!-- TODO: Replace with your deployed Streamlit Community Cloud URL -->
<!-- **[Try the Live Demo](https://your-app.streamlit.app)** -->
---
![Pipeline](https://github.com/user-attachments/assets/c74a9be2-96c7-4532-99da-a624c24910f4)
---
Results
Metric	Value
Classes	3 (Scottish Deerhound, Maltese, Bernese Mountain Dog)
Epochs	20
Optimizer	Adam (lr = 0.0001)
Loss	Categorical Crossentropy
Train/Val/Test Split	72% / 18% / 10%
<!-- TODO: Add your actual metrics after extracting from Colab -->
<!-- | Validation Accuracy | XX.X% | -->
<!-- | Test Accuracy | XX.X% | -->
<!-- TODO: Embed training curves and confusion matrix screenshots -->
<!-- ![Training Curves](assets/training_curves.png) -->
<!-- ![Confusion Matrix](assets/confusion_matrix.png) -->
> **Note:** Export your training history plots and confusion matrix from Colab, save them in an `assets/` folder, and uncomment the image lines above. Metrics and visuals are the single biggest credibility signal in ML repos.
---
Demo
Upload any dog image → click Predict → get the predicted breed and confidence score.
<!-- TODO: Add a GIF or screenshot of the app in action -->
<!-- ![App Demo](assets/demo.gif) -->
---
Why This Architecture?
This project uses a custom CNN built from scratch rather than a pretrained backbone (ResNet, MobileNet, EfficientNet). That was a deliberate choice — the goal was to understand what each convolutional layer learns, how filter sizes affect feature extraction, and where regularization matters most.
Design decisions:
Decreasing filter count (64 → 32 → 16 → 8): Forces the network to compress spatial information progressively, acting as a learned dimensionality reduction pipeline.
Mixed kernel sizes (5×5, 3×3, 7×7, 5×5): Captures both fine-grained texture (fur patterns) and broader structural features (ear shape, body proportions) at different depths.
L2 regularization on conv + dense layers: With only 3 classes and a limited dataset, overfitting is the primary risk. L2 penalizes large weights and keeps the model generalizable without aggressive dropout.
Low learning rate (0.0001): Prevents the optimizer from overshooting on a small dataset where gradient landscapes are noisy.
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
Tech Stack
Layer	Technology
Language	Python 3.13
Deep Learning	TensorFlow / Keras
Model Type	Convolutional Neural Network (CNN)
Web App	Streamlit
Data Source	Kaggle — Dog Breed Identification
Training Environment	Google Colab
Serving Environment	Local (Streamlit)
---
Project Structure
```text
dog_breed_app/
├── Dog_Breed_Prediction.ipynb   # Full training pipeline (Colab)
├── main_app.py                  # Streamlit web application
├── requirements.txt             # Project dependencies
├── dog_breed_pipeline.svg       # Architecture diagram
├── assets/                      # Training curves, confusion matrix, demo GIF
└── README.md
```
> **Note:** `dog_breed_model.keras` is excluded via `.gitignore` due to file size. Run `Dog_Breed_Prediction.ipynb` in Google Colab to regenerate it.
---
Getting Started
1 — Clone the repo
```bash
git clone https://github.com/kaboroinformatics/dog-breed-prediction.git
cd dog-breed-prediction
```
2 — Install dependencies
```bash
pip install -r requirements.txt
```
3 — Generate the model
Open `Dog_Breed_Prediction.ipynb` in Google Colab, run all cells, and download the output `dog_breed_model.keras` into the project root.
4 — Launch the app
```bash
python -m streamlit run main_app.py
```
Open http://localhost:8501 in your browser.
---
Lessons Learned
Overfitting is the default on small datasets. With only 3 classes, the model memorized training data quickly. L2 regularization and a conservative learning rate were the most effective countermeasures — more so than architectural changes.
Preprocessing matters as much as architecture. Normalizing pixel values to [0.0–1.0] and resizing to a consistent 224x224 had a measurable impact on convergence speed.
Custom CNNs teach you what pretrained models hide. Building from scratch forced me to reason about filter sizes, pooling strides, and capacity at each layer — concepts that transfer learning abstracts away.
---
Future Work
[ ] Scale to 120 breeds using transfer learning (MobileNetV2 or EfficientNet-B0) with fine-tuning
[ ] Data augmentation pipeline — rotation, flipping, color jitter to improve generalization
[ ] Deploy to Streamlit Community Cloud for a shareable live demo
[ ] Containerize with Docker for reproducible deployment
[ ] Add Grad-CAM visualizations to show which regions the model focuses on per prediction
[ ] Benchmark against pretrained baselines and document accuracy/parameter tradeoffs
---
Author
Kelvin R. Tobias
Software engineer transitioning into AI/ML with a research focus on computational biology and latent-based directed evolution.
B.S. Software Engineering — Western Governors University (2026) | M.S. AI Engineering Candidate — WGU (Target: Dec 2026)
LinkedIn | GitHub | Kelvinintech Consulting LLC
---
License
MIT License — free to use, modify, and distribute with attribution.
