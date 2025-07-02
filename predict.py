import joblib

# Load the saved model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Example input
text = ["Government plans to launch a new satellite next month."]

# Transform the input
vectorized_text = vectorizer.transform(text)

# Predict
prediction = model.predict(vectorized_text)

# Output
print("Prediction:", "REAL" if prediction[0] == 1 else "FAKE")
