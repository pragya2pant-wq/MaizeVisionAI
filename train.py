import tensorflow as tf
import os
import json
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.utils.class_weight import compute_class_weight
tf.keras.backend.clear_session()


# ---------------- DEBUG FOLDERS ----------------
print("Train folders:", sorted(os.listdir("dataset_split/train")))
print("Validation folders:", sorted(os.listdir("dataset_split/validation")))

# ---------------- CONFIG ----------------
IMG_SIZE = (128, 128)
BATCH_SIZE = 16
SEED = 42

# ---------------- LOAD DATA ----------------
train_ds = tf.keras.utils.image_dataset_from_directory(
    "dataset_split/train",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=SEED
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    "dataset_split/validation",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# ---------------- CLASS LABELS ----------------
class_names = train_ds.class_names
num_classes = len(class_names)

print("Training classes:", class_names)
print("Class mapping (INDEX → LABEL):")
for i, name in enumerate(class_names):
    print(i, "->", name)
print("Num classes:", num_classes)

os.makedirs("outputs", exist_ok=True)

with open("outputs/class_names.json", "w") as f:
    json.dump(class_names, f)

# ---------------- CLASS WEIGHTS (IMPORTANT FIX) ----------------
y_true = np.concatenate([y.numpy() for _, y in train_ds], axis=0)
print("Unique labels in dataset:", np.unique(y_true))
print("Label distribution:", np.bincount(y_true))

class_weights_array = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_true),
    y=y_true
)

classes = np.unique(y_true)

class_weights = {
    int(c): w for c, w in zip(classes, class_weights_array)
}
print("Class weights:", class_weights)

# ---------------- PERFORMANCE PIPELINE ----------------
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)

# ---------------- AUGMENTATION (IMPORTANT FIX) ----------------
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

# ---------------- MODEL ----------------
model = models.Sequential([
    tf.keras.Input(shape=(128, 128, 3)),

    data_augmentation,

    layers.Rescaling(1./255),

    layers.Conv2D(32, 3, activation=None),
    layers.BatchNormalization(),
    layers.ReLU(),
    layers.MaxPooling2D(),

    layers.Conv2D(64, 3, activation=None),
    layers.BatchNormalization(),
    layers.ReLU(),
    layers.MaxPooling2D(),

    layers.Conv2D(128, 3, activation=None),
    layers.BatchNormalization(),
    layers.ReLU(),
    layers.MaxPooling2D(),

    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ---------------- TRAIN ----------------
start = time.time()

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10,
    class_weight=class_weights   # 🔥 KEY FIX
)

end = time.time()

# ---------------- CHECK ----------------
assert model.output_shape[-1] == num_classes, (
    f"Mismatch: model outputs {model.output_shape[-1]} but dataset has {num_classes}"
)

print("Model output shape:", model.output_shape)

# ---------------- SAVE LOGS ----------------
import pandas as pd

pd.DataFrame(history.history).to_csv(
    "outputs/training_history.csv",
    index=False
)

metrics = {
    "train_accuracy": float(history.history["accuracy"][-1]),
    "val_accuracy": float(history.history["val_accuracy"][-1]),
    "train_loss": float(history.history["loss"][-1]),
    "val_loss": float(history.history["val_loss"][-1])
}

with open("outputs/final_metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

print("Training time:", round(end - start, 2), "seconds")

preds = model.predict(val_ds.take(1))
print("Sample prediction:", preds[0])
print("Argmax:", np.argmax(preds[0]))
print("Class:", class_names[np.argmax(preds[0])])

# ---------------- SAVE MODEL ----------------
model.save("models/maize_disease_model.keras")

print("Model saved successfully!")
print("Class order used:", class_names)