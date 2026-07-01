import os
import json
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.utils.class_weight import compute_class_weight

# ================= SAFETY SETTINGS =================
tf.keras.backend.clear_session()
tf.random.set_seed(42)
np.random.seed(42)

# ================= CONFIG =================
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
SEED = 42
AUTOTUNE = tf.data.AUTOTUNE

train_path = "dataset_split/train"
val_path = "dataset_split/validation"

# ================= CHECK DATA =================
print("\n🔍 Checking dataset folders...")
print("Train:", sorted(os.listdir(train_path)))
print("Validation:", sorted(os.listdir(val_path)))

# ================= LOAD DATA =================
train_ds = tf.keras.utils.image_dataset_from_directory(
    train_path,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=SEED
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    val_path,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = train_ds.class_names
num_classes = len(class_names)

print("\n📌 Classes:", class_names)
print("📌 Num classes:", num_classes)

# ================= SAFETY CHECK =================
if num_classes < 2:
    raise ValueError("Need at least 2 classes in dataset!")

# ================= OUTPUT DIRS =================
os.makedirs("outputs", exist_ok=True)
os.makedirs("models", exist_ok=True)

# ================= SAVE CLASS MAP =================
with open("outputs/vgg16_class_names.json", "w") as f:
    json.dump(class_names, f)

# ================= SPEED OPTIMIZATION =================
train_ds = train_ds.cache().prefetch(AUTOTUNE)
val_ds = val_ds.cache().prefetch(AUTOTUNE)

# ================= CLASS WEIGHTS =================
y_true = np.concatenate([y.numpy() for _, y in train_ds], axis=0)

class_weights_array = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_true),
    y=y_true
)

class_weights = dict(enumerate(class_weights_array))
print("\n⚖️ Class weights:", class_weights)

# ================= DATA AUGMENTATION =================
data_aug = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1)
])

# ================= VGG16 BASE =================
base_model = tf.keras.applications.VGG16(
    include_top=False,
    weights="imagenet",
    input_shape=(224, 224, 3)
)

base_model.trainable = False

# ================= MODEL BUILD =================
inputs = tf.keras.Input(shape=(224, 224, 3))

x = data_aug(inputs)
x = tf.keras.applications.vgg16.preprocess_input(x)
x = base_model(x, training=False)

x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(256, activation="relu")(x)
x = layers.Dropout(0.4)(x)
outputs = layers.Dense(num_classes, activation="softmax", dtype="float32")(x)

model = tf.keras.Model(inputs, outputs)

# ================= SAFE CHECKPOINT PATHS =================
checkpoint_path = "models/vgg16_checkpoint.keras"
final_model_path = "models/vgg16_final_v2.keras"

# ================= CALLBACKS =================
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        checkpoint_path,
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True
    )
]

# ================= OPTIONAL RESUME =================
if os.path.exists(checkpoint_path):
    print("\n🔁 Resuming from checkpoint...")
    model = tf.keras.models.load_model(checkpoint_path)

# ================= COMPILE =================
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ================= TRAIN STAGE 1 =================
print("\n🚀 Stage 1 training...")

start = time.time()

history1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=5,
    class_weight=class_weights,
    callbacks=callbacks
)

# ================= FINE TUNING =================
print("\n🔥 Fine-tuning last layers...")

base_model.trainable = True

for layer in base_model.layers[:-4]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

history2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=3,
    class_weight=class_weights,
    callbacks=callbacks
)

end = time.time()

# ================= SAVE FINAL MODEL =================
model.save(final_model_path)

# ================= MERGE HISTORY =================
history = {
    "accuracy": history1.history["accuracy"] + history2.history["accuracy"],
    "val_accuracy": history1.history["val_accuracy"] + history2.history["val_accuracy"],
    "loss": history1.history["loss"] + history2.history["loss"],
    "val_loss": history1.history["val_loss"] + history2.history["val_loss"],
}

with open("outputs/vgg16_metrics.json", "w") as f:
    json.dump(history, f, indent=4)

# ================= SUMMARY =================
print("\n✅ TRAINING COMPLETE")
print("⏱ Time:", round(end - start, 2), "seconds")
print("💾 Final model:", final_model_path)
print("🏁 Best checkpoint:", checkpoint_path)
print("📦 Classes:", class_names)