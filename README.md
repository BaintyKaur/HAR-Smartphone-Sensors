# 🏃 Human Activity Recognition from Smartphone Sensors

---

## 1. Project Title, Team Members & Course Details

**Project Title:** Human Activity Recognition from Smartphone Sensors

**Course:** [Your Course Name — e.g. CS4XX: Machine Learning]  
**Institution:** [Your University / College Name]  
**Semester:** [e.g. Spring 2025]  
**Submitted To:** [Professor / Instructor Name]

| Name | Student ID |
|------|-----------|
| [Member 1 Name] | [ID] |
| [Member 2 Name] | [ID] |
| [Member 3 Name] | [ID] |

---

## 2. Problem Statement & Motivation

**Human Activity Recognition (HAR)** is the task of automatically identifying what physical activity a person is performing, using only the accelerometer and gyroscope readings from a smartphone worn at the waist — no cameras, no wearables, no manual input.

### Problem Statement

> Given a 2.56-second sliding window of triaxial accelerometer and gyroscope sensor data sampled at 50 Hz, classify the user's activity into one of six categories: Walking, Walking Upstairs, Walking Downstairs, Sitting, Standing, or Laying — in a person-independent manner, i.e., generalising to subjects never seen during training.

### Motivation

Smartphones are ubiquitous, always on, and already equipped with the sensors needed for activity recognition. A reliable HAR system built on this data has immediate real-world impact:

- **Healthcare:** Remote monitoring of patient recovery and rehabilitation without clinic visits.
- **Elderly Care:** Automatic fall detection and daily activity tracking for independent living.
- **Fitness:** Hands-free workout logging on smartphones and smartwatches.
- **Sports Science:** Fine-grained athlete performance analysis during training sessions.
- **Accessibility:** Context-aware assistive apps that adapt to the user's current activity.

The key research challenge is that **static activities (Sitting vs Standing)** produce near-identical sensor signals, and models must generalise across people with different gaits, body types, and phone placements.

---

## 3. Dataset Description

### Source

**UCI HAR Dataset** — UC Irvine Machine Learning Repository  
🔗 https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones  
Introduced by Anguita et al. (2013).

### Collection Setup

| Property | Details |
|----------|---------|
| Subjects | 30 volunteers, aged 19–48 years |
| Device | Samsung Galaxy S II worn at the waist |
| Sampling Rate | 50 Hz |
| Window Size | 128 samples = 2.56 seconds |
| Window Overlap | 50% (64-sample step) |

### Size & Split

| Split | Subjects | Windows (Samples) |
|-------|:---:|:---:|
| Train | 21 | 7,352 |
| Test | 9 | 2,947 |
| **Total** | **30** | **10,299** |

The split is **person-independent** — no subject appears in both train and test sets.

### Features

Two representations are provided:

**1. Pre-computed Feature Vector (561-dim)**  
Derived from 17 time-domain and frequency-domain statistics (mean, std, energy, entropy, correlation, etc.) applied to each sensor channel. Used directly for SVM and Random Forest.

**2. Raw Inertial Signals — 9 channels × 128 timesteps**

| Channel | Description |
|---------|-------------|
| `body_acc_x/y/z` | Body linear acceleration (gravity removed), 3 axes |
| `body_gyro_x/y/z` | Angular velocity from gyroscope, 3 axes |
| `total_acc_x/y/z` | Total acceleration (body + gravity), 3 axes |

Used as `(N, 128, 9)` tensors for the LSTM model.

**3. Handcrafted Features (144-dim, extracted in this project)**  
90 time-domain statistics (mean, std, RMS, skewness, kurtosis, energy, range, percentiles × 9 channels) + 54 frequency-domain statistics (FFT magnitude, dominant frequency, spectral centroid × 9 channels).

### Class Distribution

| Activity | Type | Train | Test |
|----------|------|:---:|:---:|
| Walking | Dynamic | 1,226 | 496 |
| Walking Upstairs | Dynamic | 1,073 | 471 |
| Walking Downstairs | Dynamic | 986 | 420 |
| Sitting | Static | 1,286 | 491 |
| Standing | Static | 1,374 | 532 |
| Laying | Static | 1,407 | 537 |
| **Total** | | **7,352** | **2,947** |

The dataset is **well-balanced** — no class is significantly under-represented.

---

## 4. Methodology Overview — All Life Cycle Stages

```
Raw Sensor Data (9 channels × 128 timesteps, 50 Hz)
                        │
                        ▼
         Sliding Windows (2.56s, 50% overlap)
                        │
          ┌─────────────┴──────────────┐
          ▼                            ▼
   UCI 561-dim Features         Raw Sequences (128×9)
   + Handcrafted 144-dim              │
   (90 time + 54 freq)                ▼
          │                       LSTM Model
          ├──► SVM Classifier
          └──► Random Forest
                        │
                        ▼
         Person-Independent Evaluation
         (Accuracy · F1 · Confusion Matrix)
```

### Stage 1 — Problem Definition & Literature Review

Defined the classification objective and reviewed benchmarks:

| Paper | Method | Accuracy |
|-------|--------|:---:|
| Anguita et al. (2013) | SVM + 561 handcrafted features | ~89% |
| Ronao & Cho (2016) | CNN on raw signals | ~94.8% |
| Ordóñez & Roggen (2016) | DeepConvLSTM | ~95%+ |
| Wang et al. (2019) | Deep learning survey | 5–10% above classical ML |

**Key finding:** Deep models eliminate manual feature engineering and consistently outperform classical ML, but classical pipelines with good features remain competitive and interpretable.

### Stage 2 — Data Collection & Understanding

- Downloaded UCI HAR Dataset (~60 MB) programmatically via `urllib`.
- Confirmed zero subject overlap between train and test (person-independent split guaranteed).
- Visualised raw waveforms per activity, class distribution bar charts, and correlation heatmap of top 20 high-variance features.

### Stage 3 — Data Preprocessing & Cleaning

| Step | Details |
|------|---------|
| Missing value check | No NaN / Inf values in features or raw signals |
| Duplicate removal | Train and test sets verified clean |
| StandardScaler | Fit on train, applied to test (UCI 561-dim) |
| VarianceThreshold | Removed zero-variance features |
| Z-score normalisation | Per-channel for raw signals (train statistics only applied to test) |
| Handcrafted features | Extracted 144-dim time + frequency features from raw signals |
| Label encoding | Integer labels 0–5 for sklearn models |
| One-hot encoding | 6-class one-hot vectors for LSTM output layer |

All preprocessed arrays saved to `./preprocessed/` for reproducibility across stages.

### Stage 4 — SVM Classifier

Three SVM variants trained and compared:

| Model | Feature Set | Dims |
|-------|------------|:---:|
| Linear SVM | UCI pre-computed | 561 |
| RBF SVM (tuned) | UCI pre-computed | 561 |
| RBF SVM | Handcrafted time+freq | 144 |

Hyperparameter search: **RandomizedSearchCV** over `C ∈ {1, 10}` and `gamma ∈ {scale, 0.01}`, 3-fold stratified CV on a 30% data subset, then refitted on the full training set.

### Stage 5 — Random Forest Classifier

- Trained on UCI 561-dim features.
- Hyperparameter search over `n_estimators`, `max_depth`, `min_samples_split`.
- Feature importance analysis to identify the most discriminative signal statistics.
- Saved as `models/random_forest_model.pkl`.

### Stage 6 — LSTM Deep Learning Model

- Stacked LSTM layers trained on raw normalised `(128, 9)` sequences — no manual feature engineering.
- Training monitored via accuracy and loss curves over epochs.
- Zoomed confusion analysis generated for the hardest pair: **Sitting vs Standing**.
- Saved as `models/lstm_model.h5`.

### Stage 7 — Model Comparison & Final Evaluation

All models evaluated head-to-head on the held-out person-independent test set. Metrics: Accuracy, Macro F1, Weighted F1, per-class precision/recall, and confusion matrices.

---

## 5. Results Summary

### Literature Benchmark

| Approach | Typical Accuracy (UCI HAR) |
|----------|:---:|
| SVM — 561 handcrafted features | ~89% |
| Random Forest | ~91% |
| LSTM | ~92–93% |
| CNN (raw signals) | ~94–95% |
| CNN-LSTM hybrid | ~95–96% |

### Our Results

| Model | Accuracy | F1 Macro | F1 Weighted |
|-------|:---:|:---:|:---:|
| Linear SVM (UCI 561-dim) | `__`% | `__` | `__` |
| RBF SVM (UCI 561-dim) | `__`% | `__` | `__` |
| RBF SVM (Handcrafted 144-dim) | `__`% | `__` | `__` |
| Random Forest | `__`% | `__` | `__` |
| LSTM | `__`% | `__` | `__` |

> Replace `__` with the printed values from your notebook outputs.

### Key Findings

- **Dynamic activities** (Walking, Upstairs, Downstairs) are classified with high confidence across all models.
- **Sitting vs Standing** is the hardest pair — near-zero acceleration in both makes them difficult to separate.
- **LSTM** achieves competitive accuracy with no manual feature engineering, validating end-to-end deep learning for HAR.
- **Handcrafted 144-dim features** perform comparably to the full 561-dim pre-computed set, showing that targeted time + frequency statistics capture most discriminative information.
- **Person-independent evaluation** confirms results reflect real-world generalisation, not subject memorisation.

### Output Images

| File | Description |
|------|-------------|
| `stage5_cm_RF.png` | Confusion matrix — Random Forest |
| `stage5_rf_feature_importance.png` | Feature importances — Random Forest |
| `stage6_cm_LSTM.png` | Confusion matrix — LSTM |
| `stage6_lstm_training_curves.png` | Train/val accuracy & loss curves |
| `stage6_sitting_standing_zoom.png` | Sitting vs Standing zoomed confusion |
| `stage7_model_comparison.png` | All models compared side-by-side |

---

## 6. Screenshots of the Deployed Application

> Add your Streamlit app screenshots below. Drop the image files into a `screenshots/` folder in your repo, then they will render here automatically.

![Home Page](screenshots/app_home.png)
![Activity Prediction](screenshots/app_prediction.png)
![Confusion Matrix View](screenshots/app_confusion.png)
![Model Comparison Chart](screenshots/app_comparison.png)

---

## 7. Instructions for Setting Up and Running Locally

### Prerequisites

- Python 3.8+
- pip

### Step 1 — Clone the Repository

```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
```

### Step 2 — Install Dependencies

```bash
pip install -r requirements.txt
```

**`requirements.txt`:**

```
numpy
pandas
matplotlib
seaborn
scikit-learn
scipy
tensorflow>=2.10
joblib
streamlit
```

### Step 3 — Run the Notebooks (Google Colab — Recommended)

Run the notebooks **in order**. Each notebook auto-downloads the dataset on first run and saves outputs to `./preprocessed/` and `./models/`.

| Notebook | Stages |
|----------|--------|
| `svm_predictive_2.ipynb` | Stages 1–4 (Data, Preprocessing, SVM) |
| `random_forest_notebook.ipynb` | Stage 5 (Random Forest) |
| `lstm_notebook.ipynb` | Stage 6 (LSTM) |

### Step 4 — Run Locally (Jupyter)

```bash
jupyter notebook svm_predictive_2.ipynb
```

> **Note:** Ensure `UCI HAR Dataset/` is in the same directory as the notebooks. Preprocessed arrays from Stage 3 must exist before running Stages 4–6.

### Step 5 — Launch the Streamlit App

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`.

### Project Structure

```
har-project/
├── svm_predictive_2.ipynb           ← Stages 1–4
├── random_forest_notebook.ipynb     ← Stage 5
├── lstm_notebook.ipynb              ← Stage 6
├── app.py                           ← Streamlit app
├── requirements.txt
├── preprocessed/                    ← Auto-created by Stage 3
│   ├── X_train_vt.npy
│   ├── X_test_vt.npy
│   ├── scaler_ml.pkl
│   ├── label_encoder.pkl
│   └── ...
├── models/                          ← Auto-created by Stages 4–6
│   ├── svm_rbf_uci.pkl
│   ├── random_forest_model.pkl
│   └── lstm_model.h5
├── screenshots/                     ← App screenshots for README
└── README.md
```

---

## 8. Live Streamlit Deployment

🔗 **[Click here to open the live app](https://your-app-name.streamlit.app)**

> Replace the link above with your actual Streamlit Cloud URL after deployment.

### How to Deploy Your Own Instance

1. Push your repo (including `models/` and `requirements.txt`) to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **New app** → select your repo → set main file to `app.py`.
4. Click **Deploy** — your app will be live in under a minute.

---


