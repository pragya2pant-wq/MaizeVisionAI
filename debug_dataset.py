import tensorflow as tf
import numpy as np

IMG_SIZE = (128, 128)

val_ds = tf.keras.utils.image_dataset_from_directory(
    "dataset",
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=16,
    shuffle=False
)

print("Classes:", val_ds.class_names)

labels = []

for images, y in val_ds:
    labels.extend(y.numpy())

labels = np.array(labels)

print("Total labels:", len(labels))
print("Unique labels:", np.unique(labels))

for i in range(len(val_ds.class_names)):
    print(i, val_ds.class_names[i], np.sum(labels == i))