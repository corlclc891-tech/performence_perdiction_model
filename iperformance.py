import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

# ---------------------------------------------------------
# Step 1: Sample Dataset (Interns Data)
# ---------------------------------------------------------
data = {
    "task_completion_time_hrs": [
        12,
        25,
        15,
        40,
        10,
        30,
        18,
        35,
        11,
        22,
    ],  # Kam hrs = Acha performance
    "feedback_rating": [
        4.8,
        3.2,
        4.5,
        2.1,
        4.9,
        3.0,
        4.2,
        2.5,
        5.0,
        3.8,
    ],  # Rating 1-5
    "attendance_percentage": [
        95,
        80,
        92,
        60,
        98,
        75,
        88,
        65,
        100,
        85,
    ],  # Attendance %
    "performance_score": [
        90,
        65,
        88,
        45,
        95,
        60,
        82,
        50,
        98,
        75,
    ],  # Target Score (0-100)
}

df = pd.DataFrame(data)

# ---------------------------------------------------------
# Step 2: Features (X) aur Target (y) Separately Define Karein
# ---------------------------------------------------------
X = df[
    ["task_completion_time_hrs", "feedback_rating", "attendance_percentage"]
]
y = df["performance_score"]

# Train-Test Split (80% Training, 20% Testing)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------------------------------------
# Step 3: Train Random Forest Regressor
# ---------------------------------------------------------
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

rf_preds = rf_model.predict(X_test)

print("=" * 40)
print("   RANDOM FOREST MODEL EVALUATION")
print("=" * 40)
print(
    f"Mean Absolute Error (MAE) : {mean_absolute_error(y_test, rf_preds):.2f}"
)
print(f"R2 Score                  : {r2_score(y_test, rf_preds):.2f}\n")

# ---------------------------------------------------------
# Step 4: Train XGBoost Regressor
# ---------------------------------------------------------
xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
xgb_model.fit(X_train, y_train)

xgb_preds = xgb_model.predict(X_test)

print("=" * 40)
print("   XGBOOST MODEL EVALUATION")
print("=" * 40)
print(
    f"Mean Absolute Error (MAE) : {mean_absolute_error(y_test, xgb_preds):.2f}"
)
print(f"R2 Score                  : {r2_score(y_test, xgb_preds):.2f}\n")


# ---------------------------------------------------------
# Step 5: Helper Function - Intern Performance Classifier
# ---------------------------------------------------------
def predict_intern_status(task_time, feedback, attendance):
    input_data = pd.DataFrame(
        [[task_time, feedback, attendance]],
        columns=[
            "task_completion_time_hrs",
            "feedback_rating",
            "attendance_percentage",
        ],
    )
    score = rf_model.predict(input_data)[0]

    # Excel vs Struggle Threshold
    status = "Excel 🚀" if score >= 75 else "Struggle / Needs Help ⚠️"
    return score, status


# Testing Function with New Examples
print("=" * 40)
print("   NEW INTERNS PREDICTION TEST")
print("=" * 40)

new_interns = [
    {"name": "Intern A", "time": 13, "rating": 4.7, "attendance": 94},
    {"name": "Intern B", "time": 38, "rating": 2.3, "attendance": 65},
    {"name": "Intern C", "time": 20, "rating": 3.9, "attendance": 85},
]

for intern in new_interns:
    score, status = predict_intern_status(
        intern["time"], intern["rating"], intern["attendance"]
    )
    print(
        f"{intern['name']} -> Predicted Score: {score:.1f} | Status: {status}"
    )

# ---------------------------------------------------------
# Step 6: Visualizing Feature Importance (Graph Plotting)
# ---------------------------------------------------------
importances = rf_model.feature_importances_
features = X.columns

plt.figure(figsize=(8, 5))
sns.barplot(x=importances, y=features, palette="viridis")
plt.title("Feature Importance in Predicting Intern Performance")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.tight_layout()

print("\nGraph open ho raha hai...")
plt.show()
