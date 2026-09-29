Intern Performance Prediction using Machine Learning

📌 Project Overview

This project focuses on building a predictive Machine Learning pipeline to evaluate and forecast intern performance based on key operational and behavioral metrics. Using features like Task Completion Time, Feedback Ratings, and Attendance Records, the system predicts a final performance score and classifies interns into those who are likely to Excel or those who might Struggle and need extra guidance.

🛠️ Key Features

Data Preprocessing & Feature Engineering: Prepares intern metrics for machine learning models.

Dual Model Implementation: Trains and evaluates two popular regression algorithms:

Random Forest Regressor

XGBoost Regressor

Performance Evaluation: Evaluates model accuracy using Mean Absolute Error (MAE) and $R^2$ Score.

Performance Categorization: Classifies predicted scores into actionable business insights:

$\ge 75$ Score: Likely to Excel 🚀

$< 75$ Score: Struggle / Needs Support ⚠️

Feature Importance Visualization: Generates visual charts showing which factor impacts intern outcomes the most.

📊 Dataset Features

The model operates on three primary predictors to determine the outcome:

Feature Name

Description

Type

task_completion_time_hrs

Average time taken to complete assigned tasks (in hours)

Numerical

feedback_rating

Supervisor ratings on task quality (1.0 to 5.0)

Numerical

attendance_percentage

Intern's overall attendance rate (%)

Numerical

performance_score (Target)

Final calculated overall performance score (0 - 100)

Numerical

⚙️ Tech Stack & Requirements

Language: Python 3.8+

Development Environment: VS Code / Jupyter Notebook

Libraries Required:

pandas

numpy

scikit-learn

xgboost

matplotlib

seaborn

🚀 How to Run the Project locally in VS Code

1. Clone the Repository

git clone https://github.com/YOUR-USERNAME/intern-performance-prediction.git
cd intern-performance-prediction


2. Install Required Dependencies

Run the following command in your VS Code terminal to install all required Python packages:

pip install pandas numpy scikit-learn xgboost matplotlib seaborn


3. Run the Python Script

Execute the main script to train models and visualize results:

python intern_performance.py


📈 Sample Results & Evaluation

After running the pipeline, the system outputs metrics comparing Random Forest and XGBoost regressor performances:

========================================
   RANDOM FOREST MODEL EVALUATION
========================================
Mean Absolute Error (MAE) : 2.15
R2 Score                  : 0.94

========================================
   XGBOOST MODEL EVALUATION
========================================
Mean Absolute Error (MAE) : 2.80
R2 Score                  : 0.91


📌 Usage Example

The script includes an inference function predict_intern_status() to test new intern inputs:

# Example Usage
score, status = predict_intern_status(task_time=13, feedback=4.7, attendance=94)
print(f"Predicted Score: {score} | Outcome: {status}")


Output:

Predicted Score: 89.5 | Outcome: Excel 🚀


📄 License

This project is open-source and available under the MIT License.
