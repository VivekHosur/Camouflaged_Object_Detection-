import os
import random
import shutil

random.seed(42)

images = os.listdir("images")
images = [f for f in images if f.endswith(".jpg")]

random.shuffle(images)

split = int(0.8 * len(images))
train_imgs = images[:split]
val_imgs = images[split:]

for folder in [
    "train/images", "train/masks", "train/edges",
    "val/images", "val/masks", "val/edges"
]:
    os.makedirs(folder, exist_ok=True)

def copy_files(names, split_type):
    for name in names:
        shutil.copy("images/" + name, f"{split_type}/images/" + name)
        shutil.copy("masks/" + name.replace(".jpg", ".png"),
                    f"{split_type}/masks/" + name.replace(".jpg", ".png"))
        shutil.copy("edges/" + name.replace(".jpg", ".png"),
                    f"{split_type}/edges/" + name.replace(".jpg", ".png"))

copy_files(train_imgs, "train")
copy_files(val_imgs, "val")

print("split done")
