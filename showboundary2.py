import cv2
import os
import numpy as np

# ===== YOUR CUSTOM FOLDERS =====
img_dir = "images"
edge_dir = "edges"

# ===== GIVE IMAGE NAME MANUALLY =====
imgname = "camourflage_00010.jpg"   # 👈 change to your image name

# ===== READ IMAGE =====
img_path = os.path.join(img_dir, imgname)
img = cv2.imread(img_path)

# ===== READ EDGE =====
name, _ = os.path.splitext(imgname)
edge_path = os.path.join(edge_dir, name + ".png")
edge = cv2.imread(edge_path, 0)

if img is None or edge is None:
    print("Image or edge not found")
    exit()

# ===== RESIZE =====
img = cv2.resize(img, (256, 256))
edge = cv2.resize(edge, (256, 256))

# ===== THICKEN EDGE =====
kernel = np.ones((3,3), np.uint8)
edge = cv2.dilate(edge, kernel, iterations=1)

# ===== OVERLAY BOUNDARY =====
overlay = img.copy()
overlay[edge > 0] = [0, 0, 255]

# ===== SAVE OUTPUT =====
cv2.imwrite("custom_input_output.png", overlay)
print("DONE: custom_input_output.png created")
