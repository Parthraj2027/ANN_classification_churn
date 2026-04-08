# 🧠 Customer Churn Prediction — ANN Classification


> Predicting whether a bank customer will churn using an Artificial Neural Network — with a live Streamlit interface for real-time inference.

---

## 📌 Problem Statement

Customer churn is one of the most costly problems in banking. Acquiring a new customer costs **5–7x more** than retaining an existing one. This project builds a binary classifier to predict whether a customer is likely to leave the bank, enabling proactive retention strategies.

**Dataset:** [Churn Modelling Dataset](https://www.kaggle.com/datasets/shubh0799/churn-modelling) — 10,000 bank customers with 14 features including credit score, geography, age, balance, and product usage.

---

## 🏗️ Project Structure

```
ANN_classification_churn/
│
├── Churn_Modelling.csv          # Raw dataset
├── experiments.ipynb            # EDA, preprocessing, model training & evaluation
├── prediction.ipynb             # Inference testing on new samples
├── app.py                       # Streamlit web application
│
├── model.h5                     # Trained ANN model (saved weights)
├── label_encoder_gender.pkl     # LabelEncoder for Gender column
├── onehot_encoder_geo.pkl       # OneHotEncoder for Geography column
├── scaler.pkl                   # StandardScaler for numerical features
│
└── requirements.txt             # All dependencies
```

---

## ⚙️ End-to-End Pipeline

### 1. Data Preprocessing
| Step | Column | Method |
|------|--------|--------|
| Label Encoding | `Gender` | Male → 1, Female → 0 |
| One-Hot Encoding | `Geography` | France / Germany / Spain → binary columns |
| Feature Scaling | All numerics | StandardScaler (zero mean, unit variance) |
| Train/Test Split | — | 80% train / 20% test |

All fitted transformers are **serialized as `.pkl` files** to ensure the exact same transformations are applied at inference time in the Streamlit app — preventing train-serve skew.

### 2. Model Architecture

```
Input Layer  →  [11 features]
                     ↓
Hidden Layer 1  →  Dense(64, activation='relu')
                     ↓
Hidden Layer 2  →  Dense(32, activation='relu')
                     ↓
Output Layer  →  Dense(1, activation='sigmoid')
```

- **Loss:** Binary Cross-Entropy  
- **Optimizer:** Adam  
- **Epochs:** 100  
- **Batch Size:** 32  

### 3. Evaluation

| Metric | Score |
|--------|-------|
| Test Accuracy | **~86%** |
| Loss (final) | ~0.34 |

### 4. Deployment — Streamlit App

The `app.py` loads the saved `model.h5` and all preprocessors, accepts customer inputs via UI widgets, applies the full preprocessing pipeline, and returns a **real-time churn probability** with a binary prediction.

---

## 🚀 Running Locally

```bash
# 1. Clone the repository
git clone https://github.com/Parthraj2027/ANN_classification_churn.git
cd ANN_classification_churn

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the Streamlit app
streamlit run app.py
```

The app will open at `http://localhost:8501`

---

## 🧩 Key Technical Decisions

**Why save encoders as `.pkl` files?**  
Fitting encoders on training data and reusing the same fitted objects at inference time is critical. If you refit on new input, category ordering can change (e.g., OHE column order may shuffle), causing silent, hard-to-debug prediction errors.

**Why two hidden layers?**  
The dataset is tabular with ~11 features — a deep network would overfit. Two hidden layers with decreasing neurons (64 → 32) provide enough capacity to learn non-linear decision boundaries without overfitting on 10K samples.

---

## 📦 Dependencies

```
tensorflow
streamlit
scikit-learn
pandas
numpy
```

See `requirements.txt` for pinned versions.

---

## 👤 Author

**Parthraj** — [GitHub](https://github.com/Parthraj2027)
