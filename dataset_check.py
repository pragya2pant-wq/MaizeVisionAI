import os

dataset_path = "dataset"

print("\nImages per class:\n")

for cls in os.listdir(dataset_path):

    cls_path = os.path.join(dataset_path, cls)

    if os.path.isdir(cls_path):

        count = len(os.listdir(cls_path))

        print(f"{cls}: {count}")