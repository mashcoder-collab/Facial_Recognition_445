import os
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.decomposition import PCA

os.makedirs("data/processed_arrays", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("outputs/figures", exist_ok=True)

X_train = np.load("data/processed_arrays/X_train.npy")
X_test = np.load("data/processed_arrays/X_test.npy")
y_train = np.load("data/processed_arrays/y_train.npy")

IMG_SHAPE = (64, 64)
VARIANCE_TARGET = 0.95

pca_full = PCA(n_components=VARIANCE_TARGET, svd_solver="full", whiten=True, random_state=42)
pca_full.fit(X_train)
n_components = pca_full.n_components_

print(f"PCA selected {n_components} components to explain "
      f"{VARIANCE_TARGET*100:.0f}% of variance "
      f"(reduced from {X_train.shape[1]} raw pixels -> {n_components} features, "
      f"a {X_train.shape[1]/n_components:.1f}x reduction)")

X_train_pca = pca_full.transform(X_train)
X_test_pca = pca_full.transform(X_test)

np.save("data/processed_arrays/X_train_pca.npy", X_train_pca)
np.save("data/processed_arrays/X_test_pca.npy", X_test_pca)
joblib.dump(pca_full, "models/pca_model.pkl")

mean_face = pca_full.mean_.reshape(IMG_SHAPE)
eigenfaces = pca_full.components_.reshape((n_components, *IMG_SHAPE))

n_show = 10
fig, axes = plt.subplots(2, 6, figsize=(15, 5))
axes[0, 0].imshow(mean_face, cmap="gray")
axes[0, 0].set_title("Mean Face")
axes[0, 0].axis("off")

flat_axes = axes.flatten()
for i in range(n_show):
    ax = flat_axes[i + 1]
    ax.imshow(eigenfaces[i], cmap="gray")
    ax.set_title(f"Eigenface {i+1}")
    ax.axis("off")

for j in range(n_show + 1, len(flat_axes)):
    flat_axes[j].axis("off")

plt.suptitle("Mean Face and Top Eigenfaces (Principal Components)")
plt.tight_layout()
plt.savefig("outputs/figures/eigenfaces.png", dpi=130)
plt.close()
print("Saved outputs/figures/eigenfaces.png")

plt.figure(figsize=(7, 4))
cumsum = np.cumsum(pca_full.explained_variance_ratio_)
plt.plot(range(1, len(cumsum) + 1), cumsum, marker="o", markersize=3)
plt.axhline(0.95, color="red", linestyle="--", label="95% variance threshold")
plt.axvline(n_components, color="green", linestyle="--", label=f"{n_components} components chosen")
plt.xlabel("Number of Principal Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("PCA: How Many Components Do We Need?")
plt.legend()
plt.tight_layout()
plt.savefig("outputs/figures/pca_variance.png", dpi=130)
plt.close()
print("Saved outputs/figures/pca_variance.png")
