import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

class SimpleCODNet(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder
        self.e1 = ConvBlock(3, 32)
        self.e2 = ConvBlock(32, 64)
        self.e3 = ConvBlock(64, 128)

        self.pool = nn.MaxPool2d(2)

        # Decoder
        self.d2 = ConvBlock(128, 64)
        self.d1 = ConvBlock(64, 32)

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

        # Outputs
        self.mask_head = nn.Conv2d(32, 1, 1)
        self.edge_head = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        x1 = self.e1(x)
        x2 = self.e2(self.pool(x1))
        x3 = self.e3(self.pool(x2))

        d2 = self.up(x3)
        d2 = self.d2(d2)

        d1 = self.up(d2)
        d1 = self.d1(d1)

        return torch.sigmoid(self.mask_head(d1)), torch.sigmoid(self.edge_head(d1))

