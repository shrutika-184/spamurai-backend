# from flask import Flask, request, jsonify
# import joblib
# import pandas as pd
# import re

# # Initialize Flask app
# app = Flask(__name__)

# # Load model, vectorizer, and encoder
# model = joblib.load("rf_sms_model.pkl")
# vectorizer = joblib.load("vectorizer.pkl")
# sender_encoder = joblib.load("sender_encoder.pkl")

# # Text cleaning (same as used during preprocessing)
# def clean_text(text):
#     text = text.lower()
#     text = re.sub(r'http\S+', '', text)
#     text = re.sub(r'\W', ' ', text)
#     text = re.sub(r'\s+', ' ', text).strip()
#     return text

# @app.route('/')
# def home():
#     return "🚀 Spamurai Spam Detection API is running successfully!"

# @app.route('/predict', methods=['POST'])
# def predict():
#     data = request.get_json()

#     # Extract message and sender from JSON
#     message = data.get("message", "")
#     sender = data.get("sender", "")

#     # Clean message
#     cleaned_message = clean_text(message)
#     vectorized_msg = vectorizer.transform([cleaned_message])

#     # Encode sender (if in known list)
#     try:
#         sender_encoded = sender_encoder.transform([sender])
#     except:
#         sender_encoded = [0]  # unknown sender fallback

#     # Combine both features (sender + message)
#     import numpy as np
#     combined_features = np.hstack((vectorized_msg.toarray(), [sender_encoded]))

#     # Predict
#     prediction = model.predict(combined_features)[0]
#     result = "Spam" if prediction == 1 else "Ham"

#     return jsonify({
#         "message": message,
#         "sender": sender,
#         "prediction": result
#     })

# if __name__ == "__main__":
#     app.run(debug=True)


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
