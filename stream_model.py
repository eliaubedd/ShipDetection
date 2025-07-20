import torch
import torch.nn as nn

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        w = self.pool(x)
        w = self.fc(w)
        return x * w

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, use_se=True, residual=True, dropout=0.0):
        super().__init__()
        self.residual = residual and in_ch == out_ch
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=in_ch),  # depthwise
            nn.Conv2d(in_ch, out_ch, 1),
            nn.GroupNorm(8, out_ch),
            nn.SiLU(inplace=True),

            nn.Conv2d(out_ch, out_ch, 3, padding=1, groups=out_ch),
            nn.Conv2d(out_ch, out_ch, 1),
            nn.GroupNorm(8, out_ch),
            nn.SiLU(inplace=True),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        )
        self.se = SEBlock(out_ch) if use_se else nn.Identity()

    def forward(self, x):
        out = self.conv(x)
        out = self.se(out)
        if self.residual:
            return out + x
        return out

class BetterSegNet2(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = ConvBlock(3, 64, dropout=0.1)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ConvBlock(64, 128, dropout=0.1)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = ConvBlock(128, 256, dropout=0.2)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = ConvBlock(256, 512, dropout=0.3)
        self.pool4 = nn.MaxPool2d(2)

        self.bottleneck = ConvBlock(512, 1024, dropout=0.4)

        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = ConvBlock(1024, 512, dropout=0.3)
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = ConvBlock(512, 256, dropout=0.2)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = ConvBlock(256, 128, dropout=0.1)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = ConvBlock(128, 64, dropout=0.1)

        self.final = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        x1 = self.enc1(x)       # 64
        x2 = self.enc2(self.pool1(x1))   # 128
        x3 = self.enc3(self.pool2(x2))   # 256
        x4 = self.enc4(self.pool3(x3))   # 512
        x5 = self.bottleneck(self.pool4(x4))  # 1024

        x = self.up4(x5)
        x = self.dec4(torch.cat([x, x4], dim=1))
        x = self.up3(x)
        x = self.dec3(torch.cat([x, x3], dim=1))
        x = self.up2(x)
        x = self.dec2(torch.cat([x, x2], dim=1))
        x = self.up1(x)
        x = self.dec1(torch.cat([x, x1], dim=1))

        return self.final(x)  # logits
    
