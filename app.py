from flask import Flask, render_template, request
import pickle
import re

app = Flask(__name__)

# Load trained model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z ]", "", text)
    return text

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = ""
    confidence = 0

    if request.method == "POST":
        user_text = request.form["text"]

        # Clean input text
        cleaned_text = clean_text(user_text)

        # Convert text to vector
        vector = vectorizer.transform([cleaned_text])

        # Predict sentiment and confidence
        probabilities = model.predict_proba(vector)[0]
        prediction = model.classes_[probabilities.argmax()]
        confidence = round(probabilities.max() * 100, 2)

    return render_template(
        "index.html",
        result=prediction,
        confidence=confidence
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

