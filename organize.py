import os
import shutil

source_train = "dataset/train"
source_test = "dataset/test"

def organize(folder):
    cats_path = os.path.join(folder, "cats")
    dogs_path = os.path.join(folder, "dogs")

    # Create folders if not exist
    os.makedirs(cats_path, exist_ok=True)
    os.makedirs(dogs_path, exist_ok=True)

    for file in os.listdir(folder):
        file_path = os.path.join(folder, file)

        if os.path.isfile(file_path):
            if "cat" in file.lower():
                shutil.move(file_path, os.path.join(cats_path, file))
            elif "dog" in file.lower():
                shutil.move(file_path, os.path.join(dogs_path, file))

# Run for both train and test
organize(source_train)
organize(source_test)

print("Done organizing dataset ✅")