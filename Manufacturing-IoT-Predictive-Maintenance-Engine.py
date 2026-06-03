import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, auc
from xgboost import XGBClassifier

print("=== Manufacturing IoT Predictive Maintenance System ===")

# --------------------------------------------------
# WEEK 1: DATA GENERATION (if dataset not exists)
# --------------------------------------------------

try:
    df = pd.read_csv("data/sensor_data.csv")
    print("Dataset loaded successfully.")
except:
    print("Generating synthetic dataset...")
    
    np.random.seed(42)
    rows = 500

    temperature = np.random.normal(75, 10, rows)
    vibration = np.random.uniform(0.1, 1.0, rows)
    pressure = np.random.normal(35, 5, rows)

    failure = []
    for t, v in zip(temperature, vibration):
        if t > 85 and v > 0.7:
            failure.append(1)
        else:
            failure.append(0)

    df = pd.DataFrame({
        "temperature": temperature,
        "vibration": vibration,
        "pressure": pressure,
        "failure": failure
    })

    df.to_csv("data/sensor_data.csv", index=False)
    print("Dataset created and saved.")


# WEEK 1: FEATURE ENGINEERING


df["temp_roll_mean"] = df["temperature"].rolling(6).mean()
df["vib_roll_std"] = df["vibration"].rolling(6).std()
df["temp_lag1"] = df["temperature"].shift(1)

df = df.dropna()

print("Feature Engineering Completed.")


# WEEK 2: MODEL TRAINING


X = df.drop("failure", axis=1)
y = df["failure"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = XGBClassifier(eval_metric="logloss")
model.fit(X_train, y_train)

y_probs = model.predict_proba(X_test)[:, 1]
precision, recall, _ = precision_recall_curve(y_test, y_probs)
pr_auc = auc(recall, precision)

print("PR-AUC Score:", round(pr_auc, 3))

# Save model
joblib.dump(model, "models/model.pkl")
print("Model saved successfully.")


# WEEK 3: SHAP EXPLAINABILITY


print("Generating SHAP feature importance plot...")

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

shap.summary_plot(shap_values, X_test)

print("Explainability Completed.")

print("=== SYSTEM READY ===")


from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

model = joblib.load("models/model.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    df = pd.DataFrame([data])

    probability = model.predict_proba(df)[0][1]

    return jsonify({
        "failure_probability": float(probability)
    })

if __name__ == "__main__":
    app.run(debug=True)


from xgboost import XGBClassifier

model = XGBClassifier(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.1,
    eval_metric="logloss"
)

model.fit(X_train, y_train)
