# Implement your ResNet34_UNet model here
import torch
from torch import nn

class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(ResNetBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.down = None
        if stride!=1 or in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        x = self.block(x)
        if self.down is not None:
            identity = self.down(identity)
        x += identity
        x = self.relu(x)
        return x
    
class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_blocks):
        super(EncoderBlock, self).__init__()
        self.blocks = [ResNetBlock(in_channels, out_channels, 2)]
        for _ in range(1, n_blocks):
            self.blocks.append(ResNetBlock(out_channels, out_channels, 1))
        self.blocks = nn.Sequential(*self.blocks)

    def forward(self, x):
        out = self.blocks(x)
        return out, x 
    
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.block = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        
    def forward(self, x, skip):
        x = torch.cat([skip, x], dim=1)
        x = self.block(self.up(x))
        return x   

class ResNet34_UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResNet34_UNet, self).__init__()
        self.encoder1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.encoder2 = EncoderBlock(64, 64, 3)
        self.encoder3 = EncoderBlock(64, 128, 4)
        self.encoder4 = EncoderBlock(128, 256, 6)
        self.encoder5 = EncoderBlock(256, 512, 3)
        
        self.center = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        
        self.decoder4 = DecoderBlock(256+512, 32)
        self.decoder3 = DecoderBlock(32+256, 32)
        self.decoder2 = DecoderBlock(32+128, 32)
        self.decoder1 = DecoderBlock(32+64, 32)
        
        self.output = nn.Sequential(
            nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2, padding=0),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2, padding=0),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels, kernel_size=1)
        )
    
    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2, _ = self.encoder2(enc1)
        enc3, skip1 = self.encoder3(enc2)
        enc4, skip2 = self.encoder4(enc3)
        enc5, skip3 = self.encoder5(enc4)
        
        skip4 = enc5
        center = self.center(enc5)
        
        dec4 = self.decoder4(center, skip4)
        dec3 = self.decoder3(dec4, skip3)
        dec2 = self.decoder2(dec3, skip2)
        dec1 = self.decoder1(dec2, skip1)
        
        return self.output(dec1)