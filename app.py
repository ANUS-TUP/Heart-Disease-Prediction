"""
Heart Disease Prediction – Flask REST API  (v3)
================================================
Production-ready for Render deployment.
app.py lives at PROJECT ROOT for Render compatibility.

Engineered features are computed server-side — callers still send
only the original 13 clinical features (no API change from v1).
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(BASE_DIR, 'frontend')
MODEL_DIR    = os.path.join(BASE_DIR, 'model')
MODEL_PATH   = os.path.join(MODEL_DIR, 'heart_model.pkl')
META_PATH    = os.path.join(MODEL_DIR, 'model_meta.json')

app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path='')
CORS(app)

model = joblib.load(MODEL_PATH)
with open(META_PATH) as f:
    meta = json.load(f)

BASE_FEATURES = meta['features']
ALL_FEATURES  = meta.get('all_features', BASE_FEATURES)
THRESHOLD     = meta.get('decision_threshold', 0.5)

FEATURE_RANGES = {
    'age':      (1,    120),
    'sex':      (0,    1),
    'cp':       (0,    3),
    'trestbps': (50,   250),
    'chol':     (50,   650),
    'fbs':      (0,    1),
    'restecg':  (0,    2),
    'thalach':  (40,   250),
    'exang':    (0,    1),
    'oldpeak':  (0.0,  10.0),
    'slope':    (0,    2),
    'ca':       (0,    4),
    'thal':     (0,    3),
}


def validate_input(data: dict):
    errors, values = [], {}
    for feat in BASE_FEATURES:
        if feat not in data:
            errors.append(f"Missing field: '{feat}'")
            continue
        try:
            val = float(data[feat])
        except (ValueError, TypeError):
            errors.append(f"'{feat}' must be numeric, got: {data[feat]!r}")
            continue
        lo, hi = FEATURE_RANGES[feat]
        if not (lo <= val <= hi):
            errors.append(f"'{feat}' out of range [{lo}, {hi}], got {val}")
        values[feat] = val
    return values, errors


def add_engineered(v: dict) -> dict:
    d = dict(v)
    d['age_thalach']    = d['age'] * d['thalach']
    d['hr_reserve']     = 220 - d['age'] - d['thalach']
    d['chol_age_ratio'] = d['chol'] / (d['age'] + 1)
    d['bp_age_ratio']   = d['trestbps'] / (d['age'] + 1)
    d['exang_oldpeak']  = d['exang'] * d['oldpeak']
    d['ca_thal']        = d['ca'] * d['thal']
    d['cp_exang']       = d['cp'] * d['exang']
    d['age_sq']         = d['age'] ** 2
    d['oldpeak_sq']     = d['oldpeak'] ** 2
    d['thalach_sq']     = d['thalach'] ** 2
    d['ca_oldpeak']     = d['ca'] * d['oldpeak']
    d['thal_exang']     = d['thal'] * d['exang']
    d['high_risk_age']  = int(d['age'] > 55)
    d['severe_bp']      = int(d['trestbps'] > 140)
    d['low_hr']         = int(d['thalach'] < 130)
    d['multi_vessel']   = int(d['ca'] > 1)
    d['reversible_def'] = int(d['thal'] == 3)
    d['asymptomatic_cp']= int(d['cp'] == 0)
    return d


def make_prediction(values: dict):
    full = add_engineered(values)
    df   = pd.DataFrame([full])[ALL_FEATURES]
    prob = float(model.predict_proba(df)[0][1])
    pred = int(prob >= THRESHOLD)
    risk_level = "Low" if prob < 0.35 else "Moderate" if prob < 0.60 else "High"
    return {
        "prediction":  pred,
        "label":       "Heart Disease Detected" if pred == 1 else "No Heart Disease",
        "probability": round(prob * 100, 2),
        "risk_level":  risk_level,
        "confidence":  round(max(prob, 1 - prob) * 100, 2),
    }


@app.route('/')
def index():
    return send_from_directory(FRONTEND_DIR, 'index.html')


@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({"status": "ok", "model": meta['model_name']})


@app.route('/api/model-info', methods=['GET'])
def model_info():
    return jsonify({
        "model_name":          meta['model_name'],
        "accuracy":            meta['accuracy'],
        "roc_auc":             meta['roc_auc'],
        "f1_score":            meta['f1_score'],
        "decision_threshold":  THRESHOLD,
        "train_samples":       meta['train_samples'],
        "test_samples":        meta['test_samples'],
        "disease_prevalence":  meta['disease_prevalence'],
        "confusion_matrix":    meta['confusion_matrix'],
        "feature_importances": meta['feature_importances'],
        "features":            BASE_FEATURES,
    })


@app.route('/api/predict', methods=['POST'])
def predict():
    body = request.get_json(silent=True)
    if not body:
        return jsonify({"error": "Request body must be JSON"}), 400
    values, errors = validate_input(body)
    if errors:
        return jsonify({"errors": errors}), 422
    return jsonify({"input": values, "result": make_prediction(values)})


@app.route('/api/predict-batch', methods=['POST'])
def predict_batch():
    body = request.get_json(silent=True)
    if not isinstance(body, list):
        return jsonify({"error": "Expected a JSON array of patient records"}), 400
    results, failed = [], []
    for i, record in enumerate(body):
        values, errors = validate_input(record)
        if errors:
            failed.append({"index": i, "errors": errors})
        else:
            results.append({"index": i, "result": make_prediction(values)})
    return jsonify({"predictions": results, "errors": failed,
                    "total": len(body), "success": len(results)})


if __name__ == '__main__':
    port  = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') != 'production'
    print(f"\n  Heart Disease Prediction API  →  http://0.0.0.0:{port}\n")
    app.run(debug=debug, host='0.0.0.0', port=port)
