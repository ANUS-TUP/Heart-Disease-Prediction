# 🫀 Heart Disease Prediction — CardioScan AI

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11.9-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.1.0-000000?style=for-the-badge&logo=flask&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4.2-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Render](https://img.shields.io/badge/Deployed%20on-Render-46E3B7?style=for-the-badge&logo=render&logoColor=white)

![Accuracy](https://img.shields.io/badge/Accuracy-75.1%25-brightgreen?style=flat-square)
![ROC-AUC](https://img.shields.io/badge/ROC--AUC-0.8316-brightgreen?style=flat-square)
![F1 Score](https://img.shields.io/badge/F1--Score-0.7757-brightgreen?style=flat-square)
![Train Samples](https://img.shields.io/badge/Train%20Samples-8%2C000-blue?style=flat-square)
![Features](https://img.shields.io/badge/Features-13%20Clinical%20%2B%2018%20Engineered-orange?style=flat-square)

</div>

---

A **production-ready, end-to-end Machine Learning web application** that predicts the likelihood of heart disease based on 13 standard clinical features — enhanced with 18 engineered features for improved accuracy. Built as part of **B.Tech CSE Industrial Training** at **Techno Exponent** under the guidance of **Mr. Raihan Mistry**.

---

## 🌐 Live Application

> **https://heart-predict-abca.onrender.com**

> ⚠️ Free tier — first visit after inactivity may take ~30 seconds to wake up (spin-down behavior).

---

## 📁 Project Structure

```
Heart-Disease-Prediction/
│
├── app.py                   ← Flask REST API (at ROOT — required for Render)
├── requirements.txt         ← Pinned Python dependencies
├── render.yaml              ← Render cloud deployment configuration
├── .python-version          ← Pins Python 3.11.9 on Render
├── .gitignore               ← Excludes cache, venv, pyc files
│
├── frontend/
│   └── index.html           ← CardioScan AI — Full single-page UI
│
└── model/
    ├── train_model.py       ← Model training script with feature engineering
    ├── heart_model.pkl      ← Trained Random Forest Classifier (75.1% accuracy)
    └── model_meta.json      ← Metrics, threshold, features & importances
```

---

## 🧠 ML Model Details

### Algorithm — Random Forest Classifier

The model uses a **Random Forest Classifier** trained on 8,000 samples with:
- **21 raw + engineered features** (13 clinical + 18 derived)
- **Optimised decision threshold** of 0.45 (instead of default 0.5)
- **Stratified 80/20 train-test split** preserving class balance

### 🔬 Feature Engineering (18 New Features)

Beyond the 13 raw clinical inputs, the following features were engineered:

| Engineered Feature | Formula / Logic | Clinical Meaning |
|---|---|---|
| `age_thalach` | age × thalach | Age-heart rate interaction |
| `hr_reserve` | 220 − age − thalach | Heart rate reserve capacity |
| `chol_age_ratio` | chol / age | Cholesterol relative to age |
| `bp_age_ratio` | trestbps / age | Blood pressure relative to age |
| `exang_oldpeak` | exang × oldpeak | Combined angina + ST depression |
| `ca_thal` | ca × thal | Vessel count × thalassemia |
| `cp_exang` | cp × exang | Chest pain × angina interaction |
| `age_sq` | age² | Non-linear age effect |
| `oldpeak_sq` | oldpeak² | Non-linear ST depression |
| `thalach_sq` | thalach² | Non-linear heart rate effect |
| `ca_oldpeak` | ca × oldpeak | Vessels × ST depression |
| `thal_exang` | thal × exang | Thalassemia × angina |
| `high_risk_age` | age > 55 (binary) | Senior age risk flag |
| `severe_bp` | trestbps > 140 (binary) | Hypertension flag |
| `low_hr` | thalach < 120 (binary) | Low max heart rate flag |
| `multi_vessel` | ca > 1 (binary) | Multiple vessel disease flag |
| `reversible_def` | thal == 3 (binary) | Reversible defect flag |
| `asymptomatic_cp` | cp == 0 (binary) | Asymptomatic chest pain flag |

### 📈 Model Performance

| Metric | Score |
|---|---|
| **Accuracy** | **75.1%** |
| **ROC-AUC** | **0.8316** |
| **F1 Score** | **0.7757** |
| **Decision Threshold** | **0.45** (optimised) |
| Train Samples | 8,000 |
| Test Samples | 2,000 |
| Disease Prevalence | 54.1% |
| scikit-learn Version | 1.4.2 |

### 🎯 Top Feature Importances

| Rank | Feature | Importance |
|---|---|---|
| 1 | `cp` (Chest Pain Type) | 8.30% |
| 2 | `thal` (Thalassemia) | 4.74% |
| 3 | `ca` (Major Vessels) | 4.64% |
| 4 | `age` | 4.93% |
| 5 | `exang` (Exercise Angina) | 4.39% |

---

## 📋 Input Features (13 Clinical Parameters)

| # | Feature | Description | Valid Range |
|---|---|---|---|
| 1 | `age` | Patient age in years | 1 – 120 |
| 2 | `sex` | Sex (1 = Male, 0 = Female) | 0 – 1 |
| 3 | `cp` | Chest pain type (0=Typical Angina, 1=Atypical, 2=Non-anginal, 3=Asymptomatic) | 0 – 3 |
| 4 | `trestbps` | Resting blood pressure (mmHg) | 50 – 250 |
| 5 | `chol` | Serum cholesterol (mg/dl) | 50 – 650 |
| 6 | `fbs` | Fasting blood sugar > 120 mg/dl (1=Yes, 0=No) | 0 – 1 |
| 7 | `restecg` | Resting ECG (0=Normal, 1=ST-T abnormality, 2=LV hypertrophy) | 0 – 2 |
| 8 | `thalach` | Maximum heart rate achieved (bpm) | 40 – 250 |
| 9 | `exang` | Exercise-induced angina (1=Yes, 0=No) | 0 – 1 |
| 10 | `oldpeak` | ST depression induced by exercise | 0.0 – 10.0 |
| 11 | `slope` | Slope of peak exercise ST (0=Upsloping, 1=Flat, 2=Downsloping) | 0 – 2 |
| 12 | `ca` | Number of major vessels coloured by fluoroscopy | 0 – 4 |
| 13 | `thal` | Thalassemia (1=Normal, 2=Fixed Defect, 3=Reversible Defect) | 0 – 3 |

---

## 🔌 REST API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/health` | Server health check |
| `GET` | `/api/model-info` | Model metrics, features & importances |
| `POST` | `/api/predict` | Single patient prediction |
| `POST` | `/api/predict-batch` | Batch predictions (JSON array) |

### Example — Single Prediction Request

```bash
curl -X POST https://heart-predict-abca.onrender.com/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 63,
    "sex": 1,
    "cp": 0,
    "trestbps": 145,
    "chol": 233,
    "fbs": 1,
    "restecg": 0,
    "thalach": 150,
    "exang": 0,
    "oldpeak": 2.3,
    "slope": 0,
    "ca": 0,
    "thal": 1
  }'
```

### Example Response

```json
{
  "input": { "age": 63, "sex": 1, "cp": 0, "..." : "..." },
  "result": {
    "prediction": 1,
    "label": "Heart Disease Detected",
    "probability": 68.4,
    "risk_level": "High",
    "confidence": 68.4
  }
}
```

### Risk Level Classification

| Risk Level | Probability Range | Badge Color |
|---|---|---|
| 🟢 Low | < 35% | Green |
| 🟡 Moderate | 35% – 60% | Yellow |
| 🔴 High | > 60% | Red |

---

## 💻 Local Setup & Running

```bash
# Step 1 — Clone the repository
git clone https://github.com/ANUS-TUP/Heart-Disease-Prediction.git
cd Heart-Disease-Prediction

# Step 2 — Install dependencies
pip install -r requirements.txt

# Step 3 — (Optional) Retrain the model
python model/train_model.py

# Step 4 — Start the Flask server
python app.py

# Step 5 — Open in your browser
# http://localhost:5000
```

---

## ☁️ Render Deployment Configuration

### Settings Used

| Field | Value |
|---|---|
| **Runtime** | Python 3 |
| **Build Command** | `pip install -r requirements.txt` |
| **Start Command** | `gunicorn app:app --bind 0.0.0.0:$PORT --workers 2 --timeout 120` |
| **Python Version** | `3.11.9` (enforced via `.python-version`) |
| **Environment Variable** | `FLASK_ENV = production` |
| **Environment Variable** | `PYTHON_VERSION = 3.11.9` |

### Deployment Errors Fixed ✅

| Error Encountered | Root Cause | Fix Applied |
|---|---|---|
| `No module named 'app'` | `app.py` was inside `backend/` folder | Moved `app.py` to project **root** |
| `Cannot reach backend` | Frontend had `localhost:5000` hardcoded | Changed to relative `/api` path |
| `FileNotFoundError: heart_model.pkl` | `model/` folder missing from GitHub | Uploaded `model/` folder to repo |
| Python 3.14 auto-selected | Render defaulted to latest Python | Added `.python-version` file with `3.11.9` |
| Build hangs — compiling numpy | numpy/sklearn no wheels for Python 3.14 | Pinned Python 3.11.9 (has pre-built wheels) |
| `can't chdir to backend` | Wrong start command used `--chdir backend` | Removed `--chdir`, using root-level `app.py` |

---

## 🛠 Tech Stack

| Layer | Technology | Version |
|---|---|---|
| Language | Python | 3.11.9 |
| ML Framework | scikit-learn | 1.4.2 |
| ML Model | Random Forest Classifier | — |
| Data Processing | pandas + numpy | 2.2.2 / 1.26.4 |
| Model Storage | joblib | 1.4.2 |
| Backend API | Flask + Flask-CORS | 3.1.0 / 4.0.1 |
| WSGI Server | Gunicorn | 21.2.0 |
| Frontend | HTML5 / CSS3 / Vanilla JS | — |
| Deployment | Render | Cloud (Free Tier) |

---

## 🖥 Frontend Features

The **CardioScan AI** interface provides:

- 🫀 **Animated beating heart** logo in the header
- 📊 **Live model stats bar** — Accuracy, AUC, F1, Train size loaded from API
- 📋 **13-field clinical input form** with dropdowns and number inputs
- ✅ **Real-time validation** with red error highlighting on invalid fields
- 📈 **Animated probability gauge** (0–100%) on prediction result
- 🏷 **Risk badge** — Low / Moderate / High with colour coding
- 📉 **Feature importance chart** dynamically fetched from `/api/model-info`
- 📱 **Responsive design** — works on desktop, tablet and mobile

---

## 👨‍💻 Author

| Field | Details |
|---|---|
| **Name** | Anustup Das |
| **Roll No** | 22010333092 |
| **Reg. No** | 22013001996 |
| **Programme** | B.Tech CSE with Data Science Specialization |
| **University** | Brainware University |
| **Semester** | 7th — Year 4th |
| **Course** | Industrial Training (PROJ-CSD782) |
| **Company** | Techno Exponent |
| **Position** | AI/ML Intern |
| **Mentor** | Mr. Raihan Mistry |

---

## ⚠️ Disclaimer

This project is developed for **educational and demonstration purposes only**.  
It is **not a substitute for professional medical diagnosis**.  
Always consult a qualified healthcare provider for any medical concerns.

---

<div align="center">

Made with ❤️ by **Anustup Das** | Brainware University | 2025

</div>
