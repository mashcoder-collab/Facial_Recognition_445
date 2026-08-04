import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
from sklearn.metrics import confusion_matrix, classification_report

X_test_raw = np.load("data/processed_arrays/X_test.npy")  
X_test_pca = np.load("data/processed_arrays/X_test_pca.npy")
y_test = np.load("data/processed_arrays/y_test.npy")

with open("models/best_model_name.txt") as f:
    best_model_name = f.read().strip()
best_model = joblib.load("models/best_model.pkl")

y_pred = best_model.predict(X_test_pca)

with open("outputs/model_comparison.json") as f:
    comparison = json.load(f)

plt.figure(figsize=(7, 4.5))
names = list(comparison.keys())
accs = [comparison[n] * 100 for n in names]
colors = ["#2E86AB" if n != best_model_name else "#2ECC71" for n in names]
bars = plt.bar(names, accs, color=colors)
for bar, acc in zip(bars, accs):
    plt.text(bar.get_x() + bar.get_width() / 2, acc + 1, f"{acc:.1f}%",
              ha="center", fontweight="bold")
plt.ylabel("Test Accuracy (%)")
plt.title("Classifier Comparison on Eigenface Features (10 people, 30 test images)")
plt.ylim(0, 110)
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig("outputs/figures/model_comparison.png", dpi=130)
plt.close()

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=[f"P{i}" for i in range(10)],
            yticklabels=[f"P{i}" for i in range(10)])
plt.xlabel("Predicted Person")
plt.ylabel("True Person")
plt.title(f"Confusion Matrix -- {best_model_name}\nOverall Test Accuracy: {comparison[best_model_name]*100:.2f}%")
plt.tight_layout()
plt.savefig("outputs/figures/confusion_matrix.png", dpi=130)
plt.close()


report = classification_report(y_test, y_pred, target_names=[f"Person {i}" for i in range(10)])
with open("outputs/classification_report.txt", "w") as f:
    f.write(f"Best model: {best_model_name}\n\n")
    f.write(report)
print(report)

n_show = 15
idxs = np.random.RandomState(1).choice(len(y_test), size=n_show, replace=False)
fig, axes = plt.subplots(3, 5, figsize=(13, 8))
for ax, idx in zip(axes.flatten(), idxs):
    img = X_test_raw[idx].reshape(64, 64)
    true_label = y_test[idx]
    pred_label = y_pred[idx]
    correct = true_label == pred_label
    ax.imshow(img, cmap="gray")
    ax.set_title(f"True: P{true_label}  Pred: P{pred_label}",
                 color="green" if correct else "red", fontsize=10)
    ax.axis("off")
plt.suptitle("Sample Test Predictions (green = correct, red = wrong)")
plt.tight_layout()
plt.savefig("outputs/figures/sample_predictions.png", dpi=130)
plt.close()

print("\nSaved figures to outputs/figures/: model_comparison.png, confusion_matrix.png, sample_predictions.png")
print(f"Best model ({best_model_name}) overall test accuracy: {comparison[best_model_name]*100:.2f}%")
