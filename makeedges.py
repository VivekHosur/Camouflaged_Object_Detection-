import os
import cv2

maskdir = "masks"
edgedir = "edges"


os.makedirs(edgedir, exist_ok=True)

for name in os.listdir(maskdir):
    if name.endswith(".png"):
        mask = cv2.imread(os.path.join(maskdir, name), 0)
        edge = cv2.Canny(mask, 50, 150)
        cv2.imwrite(os.path.join(edgedir, name), edge)

print("edges done")
