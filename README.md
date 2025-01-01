# Calorie_Burn_Prediction_Model

## Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

## Loading the Dataset
Replace 'data/calorie_data.csv' with your dataset path
data = pd.read_csv('data/calorie_data.csv')

## Exploratory Data Analysis (EDA)
print(data.head())  # Display first few rows
print(data.info())  # Dataset structure
print(data.describe())  # Summary statistics

## Checking for missing values
missing_values = data.isnull().sum()
print("Missing values:\n", missing_values)

## Plotting correlations
plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()

## Data Preprocessing
Encoding categorical variables, if any
Example: Converting 'Gender' to binary values
data['Gender'] = data['Gender'].map({'Male': 1, 'Female': 0})

## Splitting features and target variable
X = data[['Age', 'Gender', 'Height', 'Weight', 'Exercise_Duration']]
y = data['Calories_Burned']

## Splitting into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

## Model Training
model = LinearRegression()
model.fit(X_train, y_train)

## Model Evaluation
Predictions
y_pred = model.predict(X_test)

## Evaluation metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R-squared Score: {r2}")

## Visualization
Plotting actual vs predicted values
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.7, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', lw=2)
plt.xlabel("Actual Calories Burned")
plt.ylabel("Predicted Calories Burned")
plt.title("Actual vs Predicted")
plt.show()

## Save the Model
import joblib
joblib.dump(model, 'models/calorie_burn_model.pkl')
print("Model saved as 'calorie_burn_model.pkl'")

## Prediction Function
def predict_calories(input_data):
    """
    Predict calorie burn based on input features.
    :param input_data: List or array [Age, Gender, Height, Weight, Exercise_Duration]
    :return: Predicted calorie burn
    """
    input_array = np.array(input_data).reshape(1, -1)
    prediction = model.predict(input_array)
    return prediction[0]

# Example usage
example_input = [25, 1, 175, 70, 30]  # Age, Gender, Height, Weight, Exercise_Duration
predicted_calories = predict_calories(example_input)
print(f"Predicted Calories Burned: {predicted_calories:.2f}")
