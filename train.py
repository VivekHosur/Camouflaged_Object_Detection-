import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from model import SimpleCODNet

# -----------------------------
# Dataset class
# -----------------------------
class CODDataset(Dataset):
    def __init__(self, root):
        self.imgs = sorted(os.listdir(os.path.join(root, "images")))
        self.root = root

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        name = self.imgs[idx]

        img = cv2.imread(os.path.join(self.root, "images", name))
        img = cv2.resize(img, (256, 256))
        img = img / 255.0
        img = torch.tensor(img).permute(2, 0, 1).float()

        mask = cv2.imread(os.path.join(self.root, "masks", name.replace(".jpg", ".png")), 0)
        mask = cv2.resize(mask, (256, 256))
        mask = mask / 255.0
        mask = torch.tensor(mask).unsqueeze(0).float()

        edge = cv2.imread(os.path.join(self.root, "edges", name.replace(".jpg", ".png")), 0)
        edge = cv2.resize(edge, (256, 256))
        edge = edge / 255.0
        edge = torch.tensor(edge).unsqueeze(0).float()

        return img, mask, edge

# -----------------------------
# Load data
# -----------------------------
train_data = CODDataset("train")
train_loader = DataLoader(train_data, batch_size=4, shuffle=True)

# -----------------------------
# Model, loss, optimizer
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

model = SimpleCODNet().to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# -----------------------------
# Training loop
# -----------------------------
epochs = 50

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for img, mask, edge in train_loader:
        img = img.to(device)
        mask = mask.to(device)
        edge = edge.to(device)

        pred_mask, pred_edge = model(img)

        loss_mask = criterion(pred_mask, mask)
        loss_edge = criterion(pred_edge, edge)

        loss = loss_mask + loss_edge

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch [{epoch+1}/{epochs}] Loss: {total_loss:.4f}")

torch.save(model.state_dict(), "codmodel.pth")
print("Training complete, model saved")
