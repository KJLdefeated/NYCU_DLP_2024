import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataloader import BufferflyMothLoader
from tqdm import tqdm
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels*4, kernel_size=1, bias=False)
        self.batch_norm3 = nn.BatchNorm2d(out_channels*4)
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.relu3 = nn.ReLU(inplace=True)
        self.downsample = None
        if stride != 1 or in_channels != out_channels*4:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels*4, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels*4)
            )

    def forward(self, x):
        identity = x.clone()
        x = self.relu1(self.batch_norm1(self.conv1(x)))
        x = self.relu2(self.batch_norm2(self.conv2(x)))
        x = self.batch_norm3(self.conv3(x))
        if self.downsample is not None:
            identity = self.downsample(identity)
        x += identity
        x = self.relu3(x)
        return x

class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers, stride=1):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x.clone()
        x = self.relu(self.batch_norm1(self.conv1(x)))
        x = self.batch_norm2(self.conv2(x))
        x += identity
        x = self.relu(x)
        return x

class ResNet50(nn.Module):
    def __init__(self, num_classes=100, lr=1e-3):
        super(ResNet50, self).__init__()
        self.lr = lr
        self.num_classes = num_classes
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            Bottleneck(64, 64, stride=1),
            Bottleneck(64*4, 64, stride=1),
            Bottleneck(64*4, 64, stride=1),
            Bottleneck(64*4, 128, stride=2),
            Bottleneck(128*4, 128, stride=2),
            Bottleneck(128*4, 128, stride=2),
            Bottleneck(128*4, 128, stride=2),
            Bottleneck(128*4, 256, stride=2),
            Bottleneck(256*4, 256, stride=2),
            Bottleneck(256*4, 256, stride=2),
            Bottleneck(256*4, 256, stride=2),
            Bottleneck(256*4, 256, stride=2),
            Bottleneck(256*4, 256, stride=2),
            Bottleneck(256*4, 512, stride=2),
            Bottleneck(512*4, 512, stride=2),
            Bottleneck(512*4, 512, stride=2),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(start_dim=1),
            nn.Linear(512*4, num_classes),
            #nn.Softmax(dim=1)
        )
        self.model.to(device)
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.99)

    def forward(self, x):
        return self.model(x)

    def train_step(self, x, y):
        self.model.train()
        self.optimizer.zero_grad()
        outputs = self.model(x)
        loss = self.criterion(outputs, y)
        acc = (outputs.argmax(1) == y).float().mean()
        loss.backward()
        self.optimizer.step()
        return loss.item(), acc.item()

    def val_step(self, x, y):
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(x)
            loss = self.criterion(outputs, y)
            acc = (outputs.argmax(1) == y).float().mean()
        return loss.item(), acc.item()

    def predict(self, x):
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(x)
        return outputs.argmax(1)

    def save(self, path):
        torch.save(self.model.state_dict(), path)
        
    def load(self, path):
        self.model.load_state_dict(torch.load(path))

def train_model(model, train_loader, val_loader, num_epochs=10):
    train_losses = []
    val_losses = []
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_acc = 0.0
        for x, y in tqdm(train_loader):
            x, y = x.to(device), y.to(device)
            loss, acc = model.train_step(x, y)
            train_loss += loss
            train_acc += acc
        train_loss /= len(train_loader)
        train_acc /= len(train_loader)

        val_loss = 0.0
        val_acc = 0.0
        for x, y in tqdm(val_loader):
            x, y = x.to(device), y.to(device)
            loss, acc = model.val_step(x, y)
            val_loss += loss
            val_acc += acc
        val_loss /= len(val_loader)
        val_acc /= len(val_loader)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        # model.scheduler.step()
        if epoch % 10 == 0:
            model.save(f"pretrained/resnet50/model_{epoch+1}.pth")
        print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
    
    np.save("pretrained/resnet50/train_losses.npy", np.array(train_losses))
    np.save("pretrained/resnet50/val_losses.npy", np.array(val_losses))
    print("Finished training") 

if __name__ == "__main__":
    model = ResNet50(lr = 1e-4)
    model = model.to(device)
    batch_size = 32
    train_loader = DataLoader(BufferflyMothLoader('./dataset/', 'train'), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(BufferflyMothLoader('./dataset/', 'valid'), batch_size=batch_size, shuffle=False)
    train_model(model, train_loader, val_loader, num_epochs=2000)