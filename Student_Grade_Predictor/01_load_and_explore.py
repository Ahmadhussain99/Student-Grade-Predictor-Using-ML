import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import joblib

# Load the dataset (using relative path from the script's directory)
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, "student-mat.csv.csv")
df = pd.read_csv(csv_path, sep=";")

print("Dataset Shape:", df.shape)         # Overall size
print("\nDataset Info:")
print(df.info())                          # Data types + nulls
print("\nBasic Statistics:")
print(df.describe())                      # Numeric stats
print("\nMissing Values:")
print(df.isnull().sum())                  # Confirm clean data


# Step3: Features and Target Selection

# Features (independent variables)
X = df[["G1", "G2", "studytime", "failures", "absences"]]

# Target (dependent variable)
y = df["G3"]

print("Features shape:", X.shape)
print("Target shape:", y.shape)

# Step4 : Train and Test

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training set size:", X_train.shape[0])
print("Test set size:", X_test.shape[0])


#📌 Step 5: Train a Linear Regression Model

# 1. Create the model
model = LinearRegression()

# 2. Train (fit) the model on the training data
model.fit(X_train, y_train)

# 3. Make predictions on the test data
y_pred = model.predict(X_test)

# 4. Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R-squared Score:", r2)

# Code to Compare & Plot

comparison = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
print(comparison.head(10))  # Show first 10 comparisons

# Plot Actual vs Predicted
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, color="blue", alpha=0.6)
plt.plot([0, 20], [0, 20], color="red", linestyle="--")  # Perfect prediction line
plt.xlabel("Actual G3 (Final Grade)")
plt.ylabel("Predicted G3")
plt.title("Actual vs Predicted Final Grades")
plt.grid(True)

# Save the plot instead of showing it (for non-GUI environments)
plt.savefig("actual_vs_predicted.png")
plt.close()


# Step 6 – Save Your Trained Model


# Save model
joblib.dump(model, "student_grade_predictor.pkl")

# Later, load model
loaded_model = joblib.load("student_grade_predictor.pkl")