import torch
import cv2
import os
import numpy as np
from model import SimpleCODNet

device = torch.device("cpu")

model = SimpleCODNet().to(device)
model.load_state_dict(torch.load("codmodel.pth", map_location=device))
model.eval()

img_dir = "val/images"
mask_dir = "val/masks"

def dice_iou(pred, gt):
    pred = (pred > 0.5).astype(np.uint8)
    gt = (gt > 0).astype(np.uint8)

    intersection = (pred & gt).sum()
    union = (pred | gt).sum()

    dice = (2 * intersection) / (pred.sum() + gt.sum() + 1e-8)
    iou = intersection / (union + 1e-8)

    return dice, iou

dice_scores = []
iou_scores = []

for name in os.listdir(img_dir)[:50]:
    img = cv2.imread(os.path.join(img_dir, name))
    mask = cv2.imread(os.path.join(mask_dir, name.replace(".jpg", ".png")), 0)

    if img is None or mask is None:
        continue

    img = cv2.resize(img, (256, 256))
    mask = cv2.resize(mask, (256, 256))

    img = img / 255.0
    img = torch.tensor(img).permute(2,0,1).unsqueeze(0).float().to(device)

    with torch.no_grad():
        mask_pred, edge_pred = model(img)
        pred = torch.sigmoid(mask_pred)
        pred = pred.squeeze().cpu().numpy()

    dice, iou = dice_iou(pred, mask)
    dice_scores.append(dice)
    iou_scores.append(iou)

print("Average Dice Score:", sum(dice_scores) / len(dice_scores))
print("Average IoU Score :", sum(iou_scores) / len(iou_scores))
