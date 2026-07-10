import numpy as np
from sklearn.model_selection import train_test_split

X = np.load("data/processed_arrays/X.npy")
y = np.load("data/processed_arrays/y.npy")

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.30,
    stratify=y,         
    random_state=42       
)

print(f"Total samples: {X.shape[0]}")
print(f"Train samples: {X_train.shape[0]}  ({X_train.shape[0]//10} images/person on average)")
print(f"Test samples:  {X_test.shape[0]}  ({X_test.shape[0]//10} images/person on average)")
print(f"Train class distribution: {np.bincount(y_train)}")
print(f"Test class distribution:  {np.bincount(y_test)}")

np.save("data/processed_arrays/X_train.npy", X_train)
np.save("data/processed_arrays/X_test.npy", X_test)
np.save("data/processed_arrays/y_train.npy", y_train)
np.save("data/processed_arrays/y_test.npy", y_test)
