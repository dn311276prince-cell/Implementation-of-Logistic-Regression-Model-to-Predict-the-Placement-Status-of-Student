# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Collect and preprocess the student placement dataset by selecting relevant features (such as CGPA, skills, internships) and the target variable (Placement Status).
	
2. Apply the Logistic Regression model, where the probability of placement is calculated using the sigmoid function:
3. Train the model by estimating the coefficients 
𝛽
β using maximum likelihood estimation on the training data.
4.Predict the placement status and evaluate the model performance using accuracy

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: dilip kumar R
RegisterNumber:  25017135
# Import required libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# ------------------------------
# Step 1: Sample dataset
# ------------------------------
data = {
    'Hours_Studied': [2, 3, 4, 5, 6, 7, 8, 9],
    'Previous_Score': [40, 50, 55, 60, 65, 70, 75, 80],
    'Internship': [0, 0, 1, 0, 1, 1, 1, 1],  # 0 = No, 1 = Yes
    'Placement': [0, 0, 0, 1, 1, 1, 1, 1]    # Target: 0 = Not Placed, 1 = Placed
}

df = pd.DataFrame(data)

# ------------------------------
# Step 2: Split into features and target
# ------------------------------
X = df[['Hours_Studied', 'Previous_Score', 'Internship']]
y = df['Placement']

# ------------------------------
# Step 3: Train-test split
# ------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# ------------------------------
# Step 4: Feature scaling
# ------------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ------------------------------
# Step 5: Create and train Logistic Regression model
# ------------------------------
model = LogisticRegression()
model.fit(X_train, y_train)

# ------------------------------
# Step 6: Make predictions
# ------------------------------
y_pred = model.predict(X_test)

# ------------------------------
# Step 7: Evaluate the model
# ------------------------------
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nAccuracy Score:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ------------------------------
# Step 8: Predict placement for a new student
# ------------------------------
new_student = np.array([[6, 68, 1]])  # Example: 6 hours studied, 68 prev score, Internship yes
new_student_scaled = scaler.transform(new_student)
placement_pred = model.predict(new_student_scaled)
placement_prob = model.predict_proba(new_student_scaled)

print(f"\nPredicted Placement Status: {'Placed' if placement_pred[0]==1 else 'Not Placed'}")
print(f"Probability of Placement: {placement_prob[0][1]:.2f}")


*/
```

## Output
Confusion Matrix:
 [[1 0]
 [0 1]]

Accuracy Score: 1.0

Classification Report:
               precision    recall  f1-score   support

           0       1.00      1.00      1.00         1
           1       1.00      1.00      1.00         1

    accuracy                           1.00         2
   macro avg       1.00      1.00      1.00         2
weighted avg       1.00      1.00      1.00         2


Predicted Placement Status: Placed
Probability of Placement: 0.78
/usr/local/lib/python3.12/dist-packages/sklearn/utils/validation.py:2739: UserWarning: X does not have valid feature names, but StandardScaler was fitted with feature names
  warnings.warn(


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
