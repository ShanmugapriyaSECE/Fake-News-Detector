# 📰 Fake News Detector

A Flask-based web application that classifies whether a news article is **real** or **fake** using Natural Language Processing (NLP) and a Machine Learning model.

## 🚀 Features
- Predicts if news content is real or fake
- Trained on a real-world dataset using TF-IDF and Logistic Regression
- Built using Flask (API) and Streamlit (UI)

## 🛠️ Tech Stack
- Python
- Flask
- Streamlit
- Scikit-learn
- Pandas, NumPy

## 📂 Files Overview
- `app.py`: Flask API to handle predictions
- `predict.py`: Core prediction logic
- `train_model.py`: Model training script
- `model.pkl`: Saved trained model
- `vectorizer.pkl`: TF-IDF vectorizer
- `streamlit_app.py`: Streamlit frontend
- `requirements.txt`: List of dependencies

## ⚙️ How to Run Locally
```bash
# 1. Clone the repo
git clone https://github.com/ShanmugapriyaSECE/Fake-News-Detector.git
cd Fake-News-Detector

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app (choose one)
python app.py          # Flask backend
streamlit run streamlit_app.py   # Streamlit frontend
```
