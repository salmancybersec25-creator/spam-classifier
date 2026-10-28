# 🛡️ Modern Scam Message Detector

A Streamlit web app that detects **modern scam messages** (SMS, WhatsApp, UPI/KYC/OTP fraud) with **99% accuracy** using an ensemble ML model.

## What it does
Paste any message → app classifies it as **SCAM** or **LEGITIMATE** with confidence score and rule-based red flags.

## AI Concept
- Soft-voting ensemble: Logistic Regression + Multinomial Naive Bayes + Calibrated LinearSVC
- Word TF-IDF (1–2 grams) + Character TF-IDF (3–5 grams) + 11 hand-crafted features
- Trained on SMS Spam Collection (~5,500 messages) augmented with 50+ modern scam samples

## Results
| Class | Precision | Recall | F1 |
|-------|-----------|--------|-----|
| Ham   | 0.99 | 1.00 | 0.99 |
| Spam  | 0.99 | 0.92 | 0.95 |
| **Accuracy** | — | — | **99%** |

## How to run
```bash
pip install -r requirements.txt
python train.py
streamlit run app.py