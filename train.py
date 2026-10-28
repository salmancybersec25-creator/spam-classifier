import pandas as pd
import joblib
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# -------------------------------------------------------------------
# 1. Load original SMS spam dataset
# -------------------------------------------------------------------
URL = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
df = pd.read_csv(URL, sep="\t", header=None, names=["label", "message"])
df["label"] = df["label"].map({"ham": 0, "spam": 1})

# -------------------------------------------------------------------
# 2. Expanded modern scam + ham examples
# -------------------------------------------------------------------
modern_scams = [
    "Dear customer, your account is blocked. Complete KYC immediately by clicking this link.",
    "Your OTP is 482910. Do not share it with anyone. If this was not you, call us.",
    "Congratulations! You have won a $500 gift card. Claim now at bit.ly/claim",
    "Your parcel is held at customs. Pay a small fee to release it: http://fake.link",
    "URGENT: Your bank account will be suspended. Verify your details here.",
    "You have received a refund of Rs. 2,500. Click here to accept.",
    "Invest $100 today and earn $1000 in 24 hours. Guaranteed returns!",
    "Your electricity bill is overdue. Pay now to avoid disconnection.",
    "Final warning: Your mobile number will be deactivated. Update your KYC.",
    "You are selected for a part-time job. Earn 5000 daily from home. WhatsApp us.",
    "Dear user, your ATM card is blocked. Call this number to reactivate.",
    "Your loan is approved. Pay processing fee to receive funds immediately.",
    "We detected suspicious login. Verify your identity: http://secure-login.xyz",
    "Claim your free recharge now. Limited offer. Click the link.",
    "Your income tax refund is ready. Submit bank details to receive it.",
    "You have an unpaid toll. Pay immediately to avoid penalty.",
    "Your Netflix subscription failed. Update payment details here.",
    "Exclusive offer: Buy 1 get 3 free. Order now before stock ends.",
    "You have 1 new voicemail. Listen here: http://voicemail-scam.com",
    "Your social security number is compromised. Call our helpline now.",
    "Dear customer, your debit card is temporarily blocked. Verify at http://bank-verify.com",
    "Your KYC is pending. Upload documents at http://kyc-update.net to avoid suspension.",
    "You won a lottery of $1,000,000. Send your bank details to claim.",
    "Urgent: Your account will be closed in 24 hours. Login here to verify.",
    "Get instant loan approval. No documents required. Apply now.",
    "Your OTP for transaction is 123456. If not you, call 1800-123-456.",
    "Dear customer, your account is temporarily locked. Verify now: http://secure-bank.co",
    "You have a pending refund of $200. Click here to claim.",
    "Limited time offer: Double your investment in 7 days. Guaranteed.",
    "Your mobile number is selected for a prize. Send your details.",
    "Final notice: Pay your outstanding bill to avoid legal action.",
    "Your package could not be delivered. Reschedule here: http://delivery-fix.com",
    "You are eligible for a government grant. Apply now.",
    "Your WhatsApp account will be banned. Verify your number here.",
    "Earn money online. Work from home. No experience needed. WhatsApp now.",
    "Your UPI account will be deactivated. Verify at http://fake-upi.com immediately.",
    "You have pending cashback of Rs. 500. Claim now before expiry.",
    "Final warning: Your SIM will be blocked in 24 hours. Update Aadhaar now.",
    "Your credit card has been used for a $2,000 transaction. If not you, call now.",
    "Dear customer, your policy is about to lapse. Renew now to avoid cancellation.",
    "Congratulations! You are the lucky winner of our weekly draw. Send details.",
    "Your bank account has been flagged for suspicious activity. Verify now.",
    "Claim your tax refund of $1,200. Click here to fill the form.",
    "Your DHL parcel is on hold. Pay customs fee at http://dhl-fake.com.",
    "Your Apple ID has been locked. Unlock now at http://apple-verify.xyz.",
    "You have 1 unread message from the bank. Login to read now.",
    "Your HMRC refund of £500 is pending. Submit bank details urgently.",
    "Your PayPal account is limited. Restore access now: http://paypal-restore.net",
    "Urgent: Verify your account within 24 hours or it will be deleted.",
    "Dear user, your wallet is temporarily locked. Complete verification at the link.",
    "Cashback of Rs. 500 credited. Click here to withdraw before expiry.",
]

modern_ham = [
    "Hey, are we still meeting for lunch tomorrow?",
    "Your OTP for login is 998877. Do not share.",
    "Your order has been shipped. Track it here: https://amazon.com/track/123",
    "Reminder: Your appointment is scheduled for tomorrow at 10 AM.",
    "Thanks for your payment. Your receipt is attached.",
    "Can you send me the address for the party?",
    "Your monthly statement is ready. Log in to view.",
    "Happy birthday! Hope you have a great day.",
    "The meeting has been moved to 3 PM.",
    "Please review the document I sent you.",
    "I'll be late by 15 minutes, traffic is bad.",
    "Did you see the match last night? Unbelievable finish!",
    "Your Uber is arriving in 3 minutes.",
    "Your flight is confirmed for 8 PM tomorrow.",
    "Can you pick up milk on the way home?",
    "Meeting link for tomorrow: https://zoom.us/j/123456",
    "Your table at the restaurant is booked for 7 PM.",
    "Please find the invoice for last month attached.",
    "Your Amazon order is out for delivery today.",
    "Hey, I transferred the money, please check.",
]

scam_df = pd.DataFrame({"message": modern_scams, "label": 1})
ham_df = pd.DataFrame({"message": modern_ham, "label": 0})
df = pd.concat([df, scam_df, ham_df], ignore_index=True)

# -------------------------------------------------------------------
# 3. Hand-crafted features
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
# 4. Build ensemble pipeline
# -------------------------------------------------------------------
features = FeatureUnion([
    ("word_tfidf", TfidfVectorizer(
        stop_words="english", ngram_range=(1, 2), min_df=1, max_df=0.95
    )),
    ("char_tfidf", TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 5), min_df=1
    )),
    ("hand", TextFeatures()),
])

ensemble = VotingClassifier(
    estimators=[
        ("lr", LogisticRegression(max_iter=1000, class_weight="balanced", C=1.0)),
        ("nb", MultinomialNB(alpha=0.5)),
        ("svm", CalibratedClassifierCV(
            LinearSVC(class_weight="balanced", max_iter=2000), cv=3
        )),
    ],
    voting="soft",
)

model = Pipeline([("features", features), ("clf", ensemble)])

# -------------------------------------------------------------------
# 5. Train + evaluate
# -------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    df["message"], df["label"],
    test_size=0.2, random_state=42, stratify=df["label"]
)

print("Training ensemble model... (may take 20-40 seconds)")
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {acc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["ham", "spam"]))

# -------------------------------------------------------------------
# 6. Sanity check on fresh, unseen modern scams
# -------------------------------------------------------------------
print("Sanity check on unseen modern scams:")
unseen = [
    "Dear user, your UPI account will be deactivated. Verify at http://fake.com",
    "You have pending cashback of Rs. 500. Claim now.",
    "Final warning: Your SIM will be blocked in 24 hours.",
    "Hey, want to grab coffee after class?",
    "Your OTP for the transaction is 778899.",
]
for msg in unseen:
    prob = model.predict_proba([msg])[0][1]
    tag = "SCAM" if prob > 0.5 else "HAM "
    print(f"  [{tag}] {prob:5.1%}  |  {msg[:55]}")

joblib.dump(model, "model.pkl")
print("\n✅ Saved improved model.pkl")