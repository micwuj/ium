import pandas as pd
import os
from sklearn.metrics import accuracy_score, f1_score, precision_score, confusion_matrix
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

test = pd.read_csv("data/test.csv")
X_test = test.drop(columns=["is_canceled"])
y_test = test["is_canceled"]

with open("data/model_columns.txt") as f:
    columns = [line.strip() for line in f.readlines()]
X_test = pd.get_dummies(X_test)
X_test = X_test.reindex(columns=columns, fill_value=0)
X_test = X_test.astype("float32")

model = load_model("data/hotel_cancel_model.h5")

y_pred_prob = model.predict(X_test).flatten()
y_pred = (y_pred_prob > 0.5).astype(int)

acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)

print("\n=== CONFUSION MATRIX ===")
print(confusion_matrix(y_test, y_pred))

plt.figure()
plt.hist(y_pred_prob, bins=20)
plt.title("Histogram prawdopodobieństw predykcji")
plt.xlabel("Prawdopodobieństwo anulowania")
plt.ylabel("Liczba próbek")
plt.savefig("data/prediction_hist.png")
plt.close()

build_number = os.getenv("BUILD_NUMBER")
if build_number is None or not build_number.isdigit():
    build_number = "1"
build_number = int(build_number)

row = pd.DataFrame([{
    "build": build_number,
    "accuracy": acc,
    "f1": f1,
    "precision": precision
}])

metrics_path = "data/metrics.csv"
if os.path.exists(metrics_path):
    old = pd.read_csv(metrics_path)
    old = old[old["build"].notna()]
    result = pd.concat([old, row], ignore_index=True)
else:
    result = row

result["build"] = pd.to_numeric(result["build"], errors="coerce")
result = result.dropna(subset=["build"])
result["build"] = result["build"].astype(int)
result = result.sort_values("build")

result.to_csv(metrics_path, index=False)

pd.DataFrame({
    "y_true": y_test,
    "y_pred": y_pred,
    "y_prob": y_pred_prob
}).to_csv("data/predictions.csv", index=False)

plt.figure()
plt.plot(result["build"], result["accuracy"], label="Accuracy")
plt.plot(result["build"], result["f1"], label="F1")
plt.plot(result["build"], result["precision"], label="Precision")
plt.xlabel("Build")
plt.ylabel("Metric Value")
plt.legend()
plt.title("Model Evaluation Over Builds")
plt.savefig("data/metrics_plot.png")
plt.close()
