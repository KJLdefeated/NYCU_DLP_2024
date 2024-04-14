import argparse
import numpy as np
import torch
from tqdm import tqdm
from VGG19 import VGG19
from ResNet50 import ResNet50
from dataloader import BufferflyMothLoader
from torch.utils.data import DataLoader
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def evaluate(mode, model, loader):
    model.eval()
    losses = 0.0
    accs = 0.0
    for x, y in tqdm(loader):
        x, y = x.to(device), y.to(device)
        loss, acc = model.val_step(x, y)
        losses += loss
        accs += acc
    losses /= len(loader)
    accs /= len(loader)
    print(f"{mode} dataset | Loss: {losses:.4f}, Acc: {accs:.4f}")

def test(model, test_loader):
    model.eval()
    test_acc = 0.0
    for x, y in tqdm(test_loader):
        x, y = x.to(device), y.to(device)
        pred = model.predict(x)
        acc = (pred == y).float().mean().item()
        test_acc += acc
    test_acc /= len(test_loader)
    print(f"Test Acc: {test_acc:.4f}")

def train(args, model, train_loader, val_loader, num_epochs=10):
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
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

        model.eval()
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
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        if args.lr_scheduler:
            model.scheduler.step()
        save_path = f"pretrained/{args.model}"
        os.makedirs(save_path, exist_ok=True)
        if epoch % 10 == 0:
            model.save(f"{save_path}/model_{args.load_model_epoch + epoch}.pth")
        print(f"Epoch {args.load_model_epoch + epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, LR: {model.scheduler.get_last_lr()[0]}")
    
    np.save(f"{save_path}/train_losses.npy", np.array(train_losses))
    np.save(f"{save_path}/val_losses.npy", np.array(val_losses))
    np.save(f"{save_path}/train_accs.npy", np.array(train_accs))
    np.save(f"{save_path}/val_accs.npy", np.array(val_accs))
    print("Finished training") 
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ButterflyMoth Classification")
    parser.add_argument("--mode", type=str, default="train", help="train, test, or evaluate")
    parser.add_argument("--model", type=str, default="R", help="R: resnet / V: vgg")
    parser.add_argument('--epochs', '-e', type=int, default=200, help='number of epochs')
    parser.add_argument('--batch_size', '-b', type=int, default=32, help='batch size')
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--lr_scheduler', '-lrs', type=bool, default=False, help='learning rate scheduler')
    parser.add_argument('--load_model_epoch', '-lme', type=int, default=0, help='load model epoch')
    args = parser.parse_args()
    if args.model == "R":
        model = ResNet50(lr = args.learning_rate)
        print("ResNet50")
    elif args.model == "V":
        model = VGG19(lr = args.learning_rate)
        print("VGG19")
    model = model.to(device)
    if args.load_model_epoch != 0:
        print(f"Loading model_{args.load_model_epoch}.pth")
        model.load(f"pretrained/{args.model}/model_{args.load_model_epoch}.pth")
    
    if args.mode == "train":
        print("Training...")
        train_loader = DataLoader(BufferflyMothLoader('./dataset/', 'train'), batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(BufferflyMothLoader('./dataset/', 'valid'), batch_size=args.batch_size, shuffle=False)
        train(args, model, train_loader, val_loader, num_epochs=args.epochs)
    elif args.mode == "test":
        test_loader = DataLoader(BufferflyMothLoader('./dataset/', 'test'), batch_size=1, shuffle=False)
        test(model, test_loader)
    elif args.mode == "evaluate":
        train_loader = DataLoader(BufferflyMothLoader('./dataset/', 'train'), batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(BufferflyMothLoader('./dataset/', 'valid'), batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(BufferflyMothLoader('./dataset/', 'test'), batch_size=1, shuffle=False)
        evaluate("train", model, train_loader)
        evaluate("valid", model, val_loader)
        evaluate("test", model, test_loader)
    else:
        print(f"Unknown mode: {args.mode}")