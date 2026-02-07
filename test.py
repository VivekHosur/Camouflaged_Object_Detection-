import os
import cv2
import torch
import numpy as np
from model import SimpleCODNet

device = "cuda" if torch.cuda.is_available() else "cpu"

# load model
model = SimpleCODNet().to(device)
model.load_state_dict(torch.load("codmodel.pth", map_location=device))
model.eval()

# load one validation image
imgname = sorted(os.listdir("val/images"))[0]
imgpath = "val/images/" + imgname

orig = cv2.imread(imgpath)
orig_resized = cv2.resize(orig, (256, 256))

img = orig_resized / 255.0
img = torch.tensor(img).permute(2, 0, 1).unsqueeze(0).float().to(device)

# inference
with torch.no_grad():
    _, pred_edge = model(img)

edge = torch.sigmoid(pred_edge)[0, 0].cpu().numpy()

# STRONG threshold
edge_bin = (edge > 0.2).astype(np.uint8) * 255

# thicken boundary
kernel = np.ones((3, 3), np.uint8)
edge_bin = cv2.dilate(edge_bin, kernel, iterations=1)

# overlay boundary in RED
overlay = orig_resized.copy()
overlay[edge_bin == 255] = [0, 0, 255]   # RED boundary

# save outputs
cv2.imwrite("boundary_overlay.png", overlay)
cv2.imwrite("pred_edge_raw.png", edge_bin)

print("Boundary image saved")
