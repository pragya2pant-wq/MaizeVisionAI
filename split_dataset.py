import os
import shutil
import random

random.seed(42)

SOURCE = "dataset"
DEST = "dataset_split"

TRAIN_RATIO = 0.8

classes = [d for d in os.listdir(SOURCE)
           if os.path.isdir(os.path.join(SOURCE, d))]

for cls in classes:

    src_folder = os.path.join(SOURCE, cls)

    images = os.listdir(src_folder)

    random.shuffle(images)

    split_index = int(len(images) * TRAIN_RATIO)

    train_images = images[:split_index]
    val_images = images[split_index:]

    train_folder = os.path.join(DEST, "train", cls)
    val_folder = os.path.join(DEST, "validation", cls)

    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(val_folder, exist_ok=True)

    for img in train_images:
        shutil.copy2(
            os.path.join(src_folder, img),
            os.path.join(train_folder, img)
        )

    for img in val_images:
        shutil.copy2(
            os.path.join(src_folder, img),
            os.path.join(val_folder, img)
        )

print("Dataset split completed.")