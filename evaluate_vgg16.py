import os
import json
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

# ================= CONFIG =================
MODEL_PATH = "models/vgg16_final_v2.keras"
DATA_PATH = "dataset_split/validation"
IMG_SIZE = (224, 224)
BATCH_SIZE = 16

# ================= LOAD MODEL =================
print("\n🔄 Loading VGG16 model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded:", MODEL_PATH)
print("Output shape:", model.output_shape)

# ================= LOAD DATA =================
print("\n📂 Loading validation dataset...")

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_PATH,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = val_ds.class_names
num_classes = len(class_names)

print("\n📌 Classes:", class_names)

os.makedirs("outputs", exist_ok=True)

with open("outputs/vgg16_class_names.json", "w") as f:
    json.dump(class_names, f)

# ================= COLLECT PREDICTIONS =================
y_true = []
y_pred = []
all_probs = []

print("\n🧪 Running predictions...")

for images, labels in val_ds:
    preds = model.predict(images, verbose=0)

    pred_labels = np.argmax(preds, axis=1)

    y_true.extend(labels.numpy())
    y_pred.extend(pred_labels)
    all_probs.extend(preds)

y_true = np.array(y_true)
y_pred = np.array(y_pred)
all_probs = np.array(all_probs)

# ================= CONFUSION MATRIX =================
cm = confusion_matrix(
    y_true,
    y_pred,
    labels=range(num_classes)
)

cm_normalized = cm.astype("float") / cm.sum(axis=1, keepdims=True)

# ================= CLASSIFICATION REPORT =================
report = classification_report(
    y_true,
    y_pred,
    target_names=class_names,
    output_dict=True
)

report_df = pd.DataFrame(report).transpose()
report_df.to_csv("outputs/vgg16_classification_report.csv")

# ================= CLASS-WISE ACCURACY =================
print("\n📊 Class-wise Accuracy:")

class_acc = cm.diagonal() / cm.sum(axis=1)

for i, acc in enumerate(class_acc):
    print(f"{class_names[i]}: {acc:.3f}")

# ================= MISCLASSIFIED ANALYSIS =================
misclassified = []

for i in range(len(y_true)):
    if y_true[i] != y_pred[i]:
        misclassified.append({
            "True": class_names[y_true[i]],
            "Predicted": class_names[y_pred[i]],
            "Confidence": float(np.max(all_probs[i]))
        })

pd.DataFrame(misclassified).to_csv("outputs/vgg16_misclassified.csv", index=False)

print("\n📁 Misclassified samples saved")

# ================= CONFIDENCE ANALYSIS =================
confidence = np.max(all_probs, axis=1)

print("\n📌 Confidence Stats:")
print("Average confidence:", np.mean(confidence))
print("Low confidence (<0.6):", np.sum(confidence < 0.6))

# ================= TOP-K CONFUSION =================
top2 = np.argsort(all_probs, axis=1)[:, -2:]

print("\n🔍 Top-2 prediction sample:")
print("Example:", top2[:5])

# ================= SAVE CONFUSION MATRIX =================
cm_df = pd.DataFrame(
    cm,
    index=class_names,
    columns=class_names
)

cm_df.to_csv("outputs/vgg16_confusion_matrix.csv")

# ================= SAVE NORMALIZED MATRIX =================
norm_df = pd.DataFrame(
    cm_normalized,
    index=class_names,
    columns=class_names
)

norm_df.to_csv("outputs/vgg16_confusion_matrix_normalized.csv")

# ================= PLOT =================
plt.figure(figsize=(8, 8))

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

disp.plot(cmap="Greens", values_format="d")

plt.title("VGG16 Confusion Matrix")
plt.savefig("outputs/vgg16_confusion_matrix.png", dpi=300, bbox_inches="tight")
plt.close()

# ================= SUMMARY =================
print("\n✅ EVALUATION COMPLETE")

print("\n📁 Files generated:")
print("- vgg16_confusion_matrix.csv")
print("- vgg16_confusion_matrix_normalized.csv")
print("- vgg16_confusion_matrix.png")
print("- vgg16_classification_report.csv")
print("- vgg16_misclassified.csv")