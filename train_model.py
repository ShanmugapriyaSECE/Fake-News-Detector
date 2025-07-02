import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
import joblib

# Sample fake news dataset (you can use your own later)
data = {
    'text': [
        "Donald Trump sends out his first tweet as president",
        "NASA confirms Earth will go dark for 6 days in November 2023",
        "COVID-19 vaccine approved by WHO",
        "Actor spotted with alien — shocking evidence!",
        "Biden signs climate change bill into law"
    ],
    'label': ['REAL', 'FAKE', 'REAL', 'FAKE', 'REAL']
}

df = pd.DataFrame(data)

# Vectorization
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X = vectorizer.fit_transform(df['text'])
y = df['label']

# Train-test split (not needed here, but good practice)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = PassiveAggressiveClassifier(max_iter=50)
model.fit(X_train, y_train)

# Save both the model and vectorizer
joblib.dump(model, 'model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')

print("✅ Model and vectorizer saved!")
