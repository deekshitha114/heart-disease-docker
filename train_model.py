
import pandas as pd
import joblib
import json

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
data = pd.read_csv("heart.csv")

# Separate input and output
X = data.drop("target", axis=1)
y = data["target"]

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train the model
model = RandomForestClassifier(
    n_estimators=100, random_state=42
)
model.fit(X_train, y_train)

# Evaluate the model
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print("Model trained successfully")
print("Accuracy:", round(accuracy * 100, 2), "%")

# Save the trained model
joblib.dump(model, "heart_model.pkl")

with open("metrics.json", "w") as f:
    json.dump({"accuracy": accuracy}, f)

print("Model saved successfully")
