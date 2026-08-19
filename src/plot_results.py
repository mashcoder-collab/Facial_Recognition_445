import matplotlib.pyplot as plt
import numpy as np


# Model names
models = ["KNN", "SVM", "Random Forest"]

# Mean accuracy from 5-fold cross-validation
mean_accuracy = [97, 99, 100]

# Standard deviation
std_accuracy = [4, 2, 0]


# Create bar chart
plt.figure(figsize=(8, 5))

bars = plt.bar(
    models,
    mean_accuracy,
    yerr=std_accuracy,
    capsize=5
)


# Add accuracy values above bars
for bar, accuracy in zip(bars, mean_accuracy):

    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.5,
        f"{accuracy}%",
        ha="center"
    )


plt.title("ML Model Comparison")
plt.xlabel("Machine Learning Model")
plt.ylabel("Mean Accuracy (%)")

plt.ylim(0, 105)

plt.tight_layout()

# Save figure
plt.savefig("../results/model_comparison.png", dpi=300)

plt.show()