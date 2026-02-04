


from flask import Flask, request, jsonify
import pickle
import re
import string
import numpy as np
from scipy.sparse import hstack

app = Flask(__name__)

# Load ML components
rf_model = pickle.load(open("rf_sms_model.pkl", "rb"))
tfidf = pickle.load(open("vectorizer.pkl", "rb"))
sender_encoder = pickle.load(open("sender_encoder.pkl", "rb"))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    return text

@app.route("/")
def home():
    return {
        "status": "Spamurai backend running",
        "endpoint": "/predict"
    }

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    message = data.get("message", "")
    sender_id = data.get("sender_id", "")
    link_present = data.get("link_present", 0)

    cleaned = clean_text(message)
    text_vec = tfidf.transform([cleaned])

    sender_encoded = (
        sender_encoder.transform([sender_id])[0]
        if sender_id in sender_encoder.classes_
        else 0
    )

    features = hstack([
        text_vec,
        np.array([[link_present, sender_encoded]])
    ])

    pred = rf_model.predict(features)[0]
    prob = rf_model.predict_proba(features)[0][pred]

    return jsonify({
        "prediction": "Spam" if pred == 1 else "Ham",
        "confidence": round(prob * 100, 2)
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

