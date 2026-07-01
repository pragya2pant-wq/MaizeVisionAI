import tensorflow as tf
import numpy as np
import pandas as pd
import json
import os
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

# -----------------------------
# CONFIG (IMPORTANT FIX)
# -----------------------------
IMG_SIZE = (224, 224)   # FIXED: must match VGG16/custom inference choice
BATCH_SIZE = 16

# -----------------------------
# LOAD MODEL
# -----------------------------
model = tf.keras.models.load_model(
    "models/maize_disease_model.keras"
)

# -----------------------------
# LOAD VALIDATION DATA
# -----------------------------
val_ds = tf.keras.utils.image_dataset_from_directory(
    "dataset_split/validation",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = val_ds.class_names

# Save class names (for app consistency)
os.makedirs("outputs", exist_ok=True)

with open("outputs/class_names.json", "w") as f:
    json.dump(class_names, f)

# -----------------------------
# NORMALIZATION (CRITICAL FIX)
# -----------------------------
# If model is VGG16-based → use preprocess_input instead of /255

from tensorflow.keras.applications.vgg16 import preprocess_input

val_ds = val_ds.map(
    lambda x, y: (preprocess_input(tf.cast(x, tf.float32)), y)
)

# -----------------------------
# COLLECT PREDICTIONS
# -----------------------------
y_true = []
y_pred = []

for images, labels in val_ds:
    preds = model.predict(images, verbose=0)
    preds_labels = np.argmax(preds, axis=1)

    y_true.extend(labels.numpy())
    y_pred.extend(preds_labels)

y_true = np.array(y_true)
y_pred = np.array(y_pred)

# -----------------------------
# CONFUSION MATRIX
# -----------------------------
cm = confusion_matrix(
    y_true,
    y_pred,
    labels=range(len(class_names))
)

cm_df = pd.DataFrame(
    cm,
    index=class_names,
    columns=class_names
)

cm_df.to_csv("outputs/confusion_matrix.csv")

# -----------------------------
# CLASSIFICATION REPORT
# -----------------------------
report = classification_report(
    y_true,
    y_pred,
    target_names=class_names,
    output_dict=True
)

report_df = pd.DataFrame(report).transpose()
report_df.to_csv("outputs/classification_report.csv")

# -----------------------------
# SAVE HEATMAP
# -----------------------------
plt.figure(figsize=(8, 8))

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

disp.plot(cmap="Greens", values_format="d")

plt.title("Confusion Matrix")
plt.savefig(
    "outputs/confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# -----------------------------
# DEBUG INFO
# -----------------------------
print("\n=== EVALUATION COMPLETE ===")
print("Classes:", class_names)
print("True label distribution:", np.bincount(y_true))
print("Pred label distribution:", np.bincount(y_pred))

print("\nSaved Files:")
print("- confusion_matrix.csv")
print("- classification_report.csv")
print("- confusion_matrix.png")