import streamlit as st
import joblib
import re
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

# -------------------------------------------------------------------
# MUST be defined before load_model() so pickle can find it
# (matches the class used in train.py)
# -------------------------------------------------------------------
class TextFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        out = []
        for text in X:
            t = text.lower()
            out.append([
                min(len(text), 300) / 300,
                min(len(text.split()), 60) / 60,
                min(text.count('!'), 5) / 5,
                min(text.count('?'), 3) / 3,
                int(bool(re.search(r'http[s]?://|www\.', t))),
                int(bool(re.search(r'\b\d{10,}\b', text))),
                int(bool(re.search(r'\b(otp|kyc|pin|password|cvv|ssn|aadhaar)\b', t))),
                int(bool(re.search(r'\b(urgent|immediately|now|warning|final|suspend|blocked|locked|deactivate)\b', t))),
                int(bool(re.search(r'\b(winner|prize|lottery|claim|won|refund|gift|cashback|reward)\b', t))),
                int(bool(re.search(r'[$£€₹]|rs\.?\s?\d', t))),
                int(text.isupper() and len(text) > 5),
            ])
        return np.array(out, dtype=float)

# -------------------------------------------------------------------
# App
# -------------------------------------------------------------------
st.set_page_config(page_title="Scam Message Detector", page_icon="🛡️")

@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

model = load_model()

st.title("🛡️ Modern Scam Message Detector")
st.caption("Paste an SMS, WhatsApp, or any message — the model will flag modern scams.")

col1, col2 = st.columns(2)
with col1:
    if st.button("📥 Try a modern scam example", use_container_width=True):
        st.session_state["msg"] = (
            "Dear customer, your account is blocked. Complete KYC immediately "
            "by clicking this link: http://secure-verify.com"
        )
with col2:
    if st.button("✅ Try a legitimate message", use_container_width=True):
        st.session_state["msg"] = "Hey, are we still meeting at 5?"

text = st.text_area(
    "Message",
    value=st.session_state.get("msg", ""),
    height=150,
    placeholder="e.g. Your OTP is 123456. Do not share. Click here to verify...",
)

if st.button("Analyse Message", type="primary"):
    if not text.strip():
        st.warning("Please enter a message.")
    else:
        proba = model.predict_proba([text])[0]
        pred = model.predict([text])[0]
        label = "scam" if pred == 1 else "legitimate"
        confidence = proba[pred]

        if label == "scam":
            st.error(f"🚨 **SCAM** — confidence {confidence:.2%}")
        else:
            st.success(f"✅ **LEGITIMATE** — confidence {confidence:.2%}")

        st.progress(float(proba[1]))
        st.caption(
            f"Scam probability: {proba[1]:.2%}  |  Legitimate probability: {proba[0]:.2%}"
        )

        suspicious = []
        if re.search(r"http[s]?://|www\.", text, re.I):
            suspicious.append("contains a URL")
        if re.search(r"\b(otp|kyc|blocked|verify|urgent|lottery|prize)\b", text, re.I):
            suspicious.append("contains high-risk keywords")
        if suspicious:
            st.info("⚠️ Additional flags: " + ", ".join(suspicious))

with st.expander("ℹ️ How it works"):
    st.write("""
    - **Model:** Soft-voting ensemble of Logistic Regression, Multinomial Naive Bayes, and Calibrated LinearSVC
    - **Features:** Word TF-IDF (1–2 grams), Character TF-IDF (3–5 grams), and 11 hand-crafted structural features
    - **Training data:** Classic SMS Spam + 50+ modern scam examples (KYC, OTP, UPI, parcel, bank block)
    - **Accuracy:** 99% on held-out test set
    """)