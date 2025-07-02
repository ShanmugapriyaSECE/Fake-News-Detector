from flask import Flask, request, jsonify
from flask_cors import CORS  # Important to avoid CORS issues
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)
CORS(app)

# Dummy model training
texts = ["This is real news", "This is fake news", "Vaccines are effective", "Aliens landed in NYC"]
labels = [1, 0, 1, 0]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)

model = LogisticRegression()
model.fit(X, labels)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    news = data.get('text', '')
    x_input = vectorizer.transform([news])
    prediction = model.predict(x_input)[0]
    return jsonify({'prediction': 'Real News' if prediction == 1 else 'Fake News'})

if __name__ == '__main__':
    app.run(debug=True)
