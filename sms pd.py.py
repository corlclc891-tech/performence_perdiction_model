import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

data = {
    "feedback": [
        "The mentorship was amazing and I learned a lot of Python skills.",
        "Management was very poor, no proper guidance was provided.",
        "Great work environment, flexible hours and friendly team.",
        "Workload was too high and stipend was delayed every month.",
        "I loved the hands-on projects and team support.",
        "No feedback given on assignments, felt lost most of the time.",
        "Excellent internship program, highly recommended!",
        "Communication with the lead was terrible and disorganized.",
    ],
    # 1 = Positive, 0 = Negative
    "sentiment": [1, 0, 1, 0, 1, 0, 1, 0],
}

df = pd.DataFrame(data)

# 2. Train / Test Split
X = df["feedback"]
y = df["sentiment"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# 3. Feature Extraction (TF-IDF Vectorizer)
# Text convert into  numerical numbers
vectorizer = TfidfVectorizer(lowercase=True, stop_words="english")
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# 4. Model Training (Logistic Regression)
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# 5. Model Evaluation
y_pred = model.predict(X_test_tfidf)
print(f"Model Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%\n")
print("Classification Report:\n", classification_report(y_test, y_pred))

#  OUTCOME:   so there is find a Unseen New Feedbacks Test  and find complex problems Identify Karain !

new_feedbacks = [
    "Mentors were not available when needed.",
    "The learning opportunities and team guidance were incredible.",
    "Too much pressure and poor communication from management.",
]

# Transform & Predict
new_tfidf = vectorizer.transform(new_feedbacks)
predictions = model.predict(new_tfidf)

print("\n--- NEW FEEDBACK PREDICTIONS & ANALYSIS ---")
for text, pred in zip(new_feedbacks, predictions):
    label = "Positive 😊" if pred == 1 else "Negative ⚠️ (Needs Improvement)"
    print(f"Feedback: '{text}'")
    print(f"Sentiment: {label}\n")