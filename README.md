# 🐾 Dog Breed Prediction — CNN + Streamlit

![Python](https://img.shields.io/badge/Python-3.13-blue?style=flat-square&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=flat-square&logo=tensorflow)
![Keras](https://img.shields.io/badge/Keras-2.x-red?style=flat-square&logo=keras)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-ff4b4b?style=flat-square&logo=streamlit)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=flat-square)

A convolutional neural network (CNN) that classifies dog breeds from images, served through an interactive Streamlit web application. Built as part of a deep learning portfolio project bridging software engineering and AI/ML engineering.

---

## Project Pipeline

![Pipeline](<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 520" width="800" height="520" font-family="Georgia, 'Times New Roman', serif">

  <!-- Background -->
  <rect width="800" height="520" fill="#0f1117" rx="16"/>

  <!-- Title -->
  <text x="400" y="38" text-anchor="middle" font-size="15" font-weight="700" fill="#e2e8f0" letter-spacing="3" font-family="'Courier New', monospace">DOG BREED CNN — PROJECT PIPELINE</text>
  <line x1="60" y1="48" x2="740" y2="48" stroke="#334155" stroke-width="0.5"/>

  <!-- ===================== ROW 1 ===================== -->

  <!-- Box 1: Image loading -->
  <rect x="60" y="68" width="180" height="64" rx="10" fill="#0d3d2e" stroke="#10b981" stroke-width="1.2"/>
  <text x="150" y="91" text-anchor="middle" font-size="12" font-weight="700" fill="#10b981" font-family="'Courier New', monospace">Image loading</text>
  <text x="150" y="109" text-anchor="middle" font-size="10" fill="#6ee7b7">224×224, normalize /255</text>
  <text x="150" y="123" text-anchor="middle" font-size="9" fill="#34d399" opacity="0.7">X_data shape: (N, 224, 224, 3)</text>

  <!-- Arrow 1→2 -->
  <line x1="240" y1="100" x2="278" y2="100" stroke="#475569" stroke-width="1.5" marker-end="url(#arr)"/>

  <!-- Box 2: Label binarize -->
  <rect x="278" y="68" width="180" height="64" rx="10" fill="#0d3d2e" stroke="#10b981" stroke-width="1.2"/>
  <text x="368" y="91" text-anchor="middle" font-size="12" font-weight="700" fill="#10b981" font-family="'Courier New', monospace">Label binarize</text>
  <text x="368" y="109" text-anchor="middle" font-size="10" fill="#6ee7b7">One-hot encode breeds</text>
  <text x="368" y="123" text-anchor="middle" font-size="9" fill="#34d399" opacity="0.7">'maltese_dog' → [0, 1, 0]</text>

  <!-- Arrow 2→3 -->
  <line x1="458" y1="100" x2="496" y2="100" stroke="#475569" stroke-width="1.5" marker-end="url(#arr)"/>

  <!-- Box 3: Train/val/test split -->
  <rect x="496" y="68" width="180" height="64" rx="10" fill="#0d3d2e" stroke="#10b981" stroke-width="1.2"/>
  <text x="586" y="91" text-anchor="middle" font-size="12" font-weight="700" fill="#10b981" font-family="'Courier New', monospace">Train/val/test split</text>
  <text x="586" y="109" text-anchor="middle" font-size="10" fill="#6ee7b7">72% / 18% / 10%</text>
  <text x="586" y="123" text-anchor="middle" font-size="9" fill="#34d399" opacity="0.7">sklearn train_test_split</text>

  <!-- Arrow Row1 → Row2 (from box 2 down) -->
  <line x1="368" y1="132" x2="368" y2="168" stroke="#475569" stroke-width="1.5" marker-end="url(#arr)"/>

  <!-- ===================== ROW 2 ===================== -->

  <!-- Box 4: CNN layers -->
  <rect x="60" y="168" width="180" height="64" rx="10" fill="#1e1b4b" stroke="#818cf8" stroke-width="1.2"/>
  <text x="150" y="191" text-anchor="middle" font-size="12" font-weight="700" fill="#818cf8" font-family="'Courier New', monospace">CNN layers</text>
  <text x="150" y="209" text-anchor="middle" font-size="10" fill="#a5b4fc">4× Conv2D + MaxPool2D</text>
  <text x="150" y="223" text-anchor="middle" font-size="9" fill="#c7d2fe" opacity="0.7">filters: 64→32→16→8</text>

  <!-- Arrow 4→5 -->
  <line x1="240" y1="200" x2="278" y2="200" stroke="#475569" stroke-width="1.5" marker-end="url(#arr)"/>

  <!-- Box 5: Flatten -->
  <rect x="278" y="168" width="180" height="64" rx="10" fill="#1e1b4b" stroke="#818cf8" stroke-width="1.2"/>
  <text x="368" y="191" text-anchor="middle" font-size="12" font-weight="700" fill="#818cf8" font-family="'Courier New', monospace">Flatten</text>
  <text x="368" y="209" text-anchor="middle" font-size="10" fill="#a5b4fc">3D tensor → 1D vector</text>
  <text x="368" y="223" text-anchor="middle" font-size="9" fill="#c7d2fe" opacity="0.7">~512 values unrolled</text>

  <!-- Arrow 5→6 -->
  <line x1="458" y1="200" x2="496" y2="200" stroke="#475569" stroke-width="1.5" marker-end="url(#arr)"/>

  <!-- Box 6: Dense + Softmax -->
  <rect x="496" y="168" width="180" height="64" rx="10" fill="#1e1b4b" stroke="#818cf8" stroke-width="1.2"/>
  <text x="586" y="191" text-anchor="middle" font-size="12" font-weight="700" fill="#818cf8" font-family="'Courier New', monospace">Dense + Softmax</text>
  <text x="586" y="209" text-anchor="middle" font-size="10" fill="#a5b4fc">3-class probabilities</text>
  <text x="586" y="223" text-anchor="middle" font-size="9" fill="#c7d2fe" opacity="0.7">[0.92, 0.05, 0.03]</text>

  <!-- Arrow Row2 → Row3 -->
  <line x1="368" y1="232" x2="368" y2="268" stroke="#475569" stroke-width="1.5" marker-end="url(#arr)"/>

  <!-- ===================== ROW 3 ===================== -->

  <!-- Box 7: Training loop -->
  <rect x="60" y="268" width="180" height="64" rx="10" fill="#2d1b00" stroke="#f59e0b" stroke-width="1.2"/>
  <text x="150" y="291" text-anchor="middle" font-size="12" font-weight="700" fill="#f59e0b" font-family="'Courier New', monospace">Training loop</text>
  <text x="150" y="309" text-anchor="middle" font-size="10" fill="#fcd34d">100 epochs, batch 128</text>
  <text x="150" y="323" text-anchor="middle" font-size="9" fill="#fde68a" opacity="0.7">Adam lr=0.0001</text>

  <!-- Arrow 7→8 -->
  <line x1="240" y1="300" x2="278" y2="300" stroke="#475569" stroke-width="1.5" marker-end="url(#arr)"/>

  <!-- Box 8: Validation monitor -->
  <rect x="278" y="268" width="180" height="64" rx="10" fill="#2d1b00" stroke="#f59e0b" stroke-width="1.2"/>
  <text x="368" y="291" text-anchor="middle" font-size="12" font-weight="700" fill="#f59e0b" font-family="'Courier New', monospace">Validation monitor</text>
  <text x="368" y="309" text-anchor="middle" font-size="10" fill="#fcd34d">Check train vs val accuracy</text>
  <text x="368" y="323" text-anchor="middle" font-size="9" fill="#fde68a" opacity="0.7">97% train / 70% val</text>

  <!-- Arrow 8→9 -->
  <line x1="458" y1="300" x2="496" y2="300" stroke="#475569" stroke-width="1.5" marker-end="url(#arr)"/>

  <!-- Box 9: Save model -->
  <rect x="496" y="268" width="180" height="64" rx="10" fill="#2d1b00" stroke="#f59e0b" stroke-width="1.2"/>
  <text x="586" y="291" text-anchor="middle" font-size="12" font-weight="700" fill="#f59e0b" font-family="'Courier New', monospace">Save model</text>
  <text x="586" y="309" text-anchor="middle" font-size="10" fill="#fcd34d">.h5 / SavedModel</text>
  <text x="586" y="323" text-anchor="middle" font-size="9" fill="#fde68a" opacity="0.7">model.save('model.h5')</text>

  <!-- Arrow Row3 → Row4 -->
  <line x1="368" y1="332" x2="368" y2="368" stroke="#475569" stroke-width="1.5" marker-end="url(#arr)"/>

  <!-- ===================== ROW 4 ===================== -->

  <!-- Box 10: Flask loads model -->
  <rect x="60" y="368" width="180" height="64" rx="10" fill="#2d0a0a" stroke="#f87171" stroke-width="1.2"/>
  <text x="150" y="391" text-anchor="middle" font-size="12" font-weight="700" fill="#f87171" font-family="'Courier New', monospace">Flask app loads model</text>
  <text x="150" y="409" text-anchor="middle" font-size="10" fill="#fca5a5">app.py + model.h5</text>
  <text x="150" y="423" text-anchor="middle" font-size="9" fill="#fecaca" opacity="0.7">load_model() on startup</text>

  <!-- Arrow 10→11 -->
  <line x1="240" y1="400" x2="278" y2="400" stroke="#475569" stroke-width="1.5" marker-end="url(#arr)"/>

  <!-- Box 11: Preprocess upload -->
  <rect x="278" y="368" width="180" height="64" rx="10" fill="#2d0a0a" stroke="#f87171" stroke-width="1.2"/>
  <text x="368" y="391" text-anchor="middle" font-size="12" font-weight="700" fill="#f87171" font-family="'Courier New', monospace">Preprocess upload</text>
  <text x="368" y="409" text-anchor="middle" font-size="10" fill="#fca5a5">Resize → array → /255</text>
  <text x="368" y="423" text-anchor="middle" font-size="9" fill="#fecaca" opacity="0.7">must match training pipeline</text>

  <!-- Arrow 11→12 -->
  <line x1="458" y1="400" x2="496" y2="400" marker-end="url(#arr)" stroke="#475569" stroke-width="1.5"/>

  <!-- Box 12: Return breed -->
  <rect x="496" y="368" width="180" height="64" rx="10" fill="#2d0a0a" stroke="#f87171" stroke-width="1.2"/>
  <text x="586" y="391" text-anchor="middle" font-size="12" font-weight="700" fill="#f87171" font-family="'Courier New', monospace">Return breed</text>
  <text x="586" y="409" text-anchor="middle" font-size="10" fill="#fca5a5">argmax → class name</text>
  <text x="586" y="423" text-anchor="middle" font-size="9" fill="#fecaca" opacity="0.7">CLASS_NAMES[np.argmax(pred)]</text>

  <!-- ===================== ROW LABELS ===================== -->
  <text x="26" y="104" text-anchor="middle" font-size="8" fill="#64748b" font-family="'Courier New', monospace" transform="rotate(-90, 26, 104)">DATA PREP</text>
  <text x="26" y="204" text-anchor="middle" font-size="8" fill="#64748b" font-family="'Courier New', monospace" transform="rotate(-90, 26, 204)">MODEL</text>
  <text x="26" y="304" text-anchor="middle" font-size="8" fill="#64748b" font-family="'Courier New', monospace" transform="rotate(-90, 26, 304)">TRAINING</text>
  <text x="26" y="404" text-anchor="middle" font-size="8" fill="#64748b" font-family="'Courier New', monospace" transform="rotate(-90, 26, 404)">FLASK APP</text>

  <!-- Divider lines between rows -->
  <line x1="44" y1="158" x2="756" y2="158" stroke="#1e293b" stroke-width="0.5" stroke-dasharray="4 4"/>
  <line x1="44" y1="258" x2="756" y2="258" stroke="#1e293b" stroke-width="0.5" stroke-dasharray="4 4"/>
  <line x1="44" y1="358" x2="756" y2="358" stroke="#1e293b" stroke-width="0.5" stroke-dasharray="4 4"/>

  <!-- Footer -->
  <text x="400" y="500" text-anchor="middle" font-size="9" fill="#334155" font-family="'Courier New', monospace">kelvinintech — Dog Breed CNN Project · Keras + TensorFlow + Flask</text>

  <!-- Arrow marker -->
  <defs>
    <marker id="arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M2 1L8 5L2 9" fill="none" stroke="#475569" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
    </marker>
  </defs>

</svg>
![dog_breed_pipeline (1)](https://github.com/user-attachments/assets/c74a9be2-96c7-4532-99da-a624c24910f4)
)

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
