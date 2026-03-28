# 🐾 Dog Breed Prediction — CNN + Streamlit

![Python](https://img.shields.io/badge/Python-3.13-blue?style=flat-square&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=flat-square&logo=tensorflow)
![Keras](https://img.shields.io/badge/Keras-2.x-red?style=flat-square&logo=keras)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-ff4b4b?style=flat-square&logo=streamlit)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=flat-square)

A convolutional neural network (CNN) that classifies dog breeds from images, served through an interactive Streamlit web application. Built as part of a deep learning portfolio project bridging software engineering and AI/ML engineering.

---


![Pipeline]![Pipeline](https://github.com/user-attachments/assets/c74a9be2-96c7-4532-99da-a624c24910f4)
---

## Demo

Upload any dog image → click **Predict** → get breed + confidence score.

**Supported breeds:**
- Scottish Deerhound
- Maltese Dog
- Bernese Mountain Dog

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.13 |
| Deep Learning | TensorFlow / Keras |
| Model Type | Convolutional Neural Network (CNN) |
| Web App | Streamlit |
| Data Source | Kaggle — Dog Breed ID Competition |
| Environment | Google Colab (training) + Local (serving) |

---

## Model Architecture

```
Input (224×224×3)
→ Conv2D(64, 5×5, relu) + MaxPool2D
→ Conv2D(32, 3×3, relu, L2) + MaxPool2D
→ Conv2D(16, 7×7, relu, L2) + MaxPool2D
→ Conv2D(8,  5×5, relu, L2) + MaxPool2D
→ Flatten
→ Dense(128, relu, L2)
→ Dense(64,  relu, L2)
→ Dense(3, softmax)
```

**Training config:**
- Loss: `categorical_crossentropy`
- Optimizer: `Adam(lr=0.0001)`
- Epochs: 20
- Batch size: 128
- Train/Val/Test split: 72% / 18% / 10%

---

## Project Structure

```
dog_breed_app/
├── Dog_Breed_Prediction.ipynb   # Full training pipeline
├── main_app.py                  # Streamlit web application
├── requirements.txt             # Project dependencies
├── dog_breed_pipeline.svg       # Architecture diagram
└── README.md
```

> **Note:** `dog_breed_model.keras` is not committed due to file size.
> Run `Dog_Breed_Prediction.ipynb` in Google Colab to regenerate it.

---

## Getting Started

### 1 — Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/dog-breed-prediction.git
cd dog-breed-prediction
```

### 2 — Install dependencies
```bash
pip install -r requirements.txt
```

### 3 — Generate the model
Run `Dog_Breed_Prediction.ipynb` in Google Colab and download the output `dog_breed_model.keras` into this folder.

### 4 — Run the app
```bash
python -m streamlit run main_app.py
```

Open `http://localhost:8501` in your browser.

---

## Requirements

```
streamlit
tensorflow
numpy
Pillow
keras
```

---

## Key Concepts Demonstrated

- **CNN feature extraction** — 4 convolutional blocks learning progressively abstract visual features
- **L2 regularization** — preventing overfitting on a small dataset
- **One-hot encoding** — converting categorical breed labels to model-compatible vectors
- **Image normalization** — scaling pixel values [0–255] → [0.0–1.0]
- **Softmax output** — multi-class probability distribution across breed classes
- **Model serialization** — saving and loading trained weights in `.keras` format
- **End-to-end ML pipeline** — from raw Kaggle dataset to deployed web application

---

## Author

**Kelvin R. Tobias**
B.S. Software Engineering — Western Governors University (2026)
M.S. AI Engineering Candidate — WGU (Target: Dec 2026)

[LinkedIn](https://linkedin.com/in/YOUR_LINKEDIN) • [GitHub](https://github.com/YOUR_USERNAME) • [Kelvinintech Consulting LLC](https://kelvinintech.com)

---

## License

MIT License — free to use, modify, and distribute with attribution.
