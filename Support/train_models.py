import numpy as np
import joblib
import json
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

X_train = np.load("data/processed_arrays/X_train_pca.npy")
X_test = np.load("data/processed_arrays/X_test_pca.npy")
y_train = np.load("data/processed_arrays/y_train.npy")
y_test = np.load("data/processed_arrays/y_test.npy")

results = {}
trained_models = {}

print("Training SVM (RBF) with GridSearchCV ...")
param_grid = {
    "C": [1, 10, 50, 100, 500, 1000],
    "gamma": [0.0001, 0.001, 0.01, 0.1],
}
svm_grid = GridSearchCV(
    SVC(kernel="rbf", class_weight="balanced", probability=True, random_state=42),
    param_grid,
    cv=3,           
    n_jobs=-1,
)
svm_grid.fit(X_train, y_train)
best_svm = svm_grid.best_estimator_
svm_pred = best_svm.predict(X_test)
results["SVM (RBF)"] = accuracy_score(y_test, svm_pred)
trained_models["SVM (RBF)"] = best_svm
print(f"  Best params: {svm_grid.best_params_}")
print(f"  Test accuracy: {results['SVM (RBF)']:.4f}")

print("Training KNN ...")
knn_grid = GridSearchCV(
    KNeighborsClassifier(),
    {"n_neighbors": [1, 3, 5], "weights": ["uniform", "distance"]},
    cv=3,
    n_jobs=-1,
)
knn_grid.fit(X_train, y_train)
best_knn = knn_grid.best_estimator_
knn_pred = best_knn.predict(X_test)
results["KNN"] = accuracy_score(y_test, knn_pred)
trained_models["KNN"] = best_knn
print(f"  Best params: {knn_grid.best_params_}")
print(f"  Test accuracy: {results['KNN']:.4f}")

print("Training Logistic Regression ...")
logreg = LogisticRegression(max_iter=5000)
logreg.fit(X_train, y_train)
logreg_pred = logreg.predict(X_test)
results["Logistic Regression"] = accuracy_score(y_test, logreg_pred)
trained_models["Logistic Regression"] = logreg
print(f"  Test accuracy: {results['Logistic Regression']:.4f}")

print("Training Random Forest ...")
rf = RandomForestClassifier(n_estimators=300, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
results["Random Forest"] = accuracy_score(y_test, rf_pred)
trained_models["Random Forest"] = rf
print(f"  Test accuracy: {results['Random Forest']:.4f}")

best_model_name = max(results, key=results.get)
best_model = trained_models[best_model_name]
print(f"\nBest model: {best_model_name} (test accuracy = {results[best_model_name]:.4f})")

joblib.dump(best_model, "models/best_model.pkl")
with open("models/best_model_name.txt", "w") as f:
    f.write(best_model_name)

with open("outputs/model_comparison.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nFull comparison:")
for name, acc in sorted(results.items(), key=lambda kv: -kv[1]):
    print(f"  {name:22s}: {acc:.4f}")

# Save predictions of the best model for the evaluation step
np.save("outputs/best_model_predictions.npy",
        best_model.predict(X_test))
