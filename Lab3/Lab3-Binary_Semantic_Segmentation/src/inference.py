import argparse
import torch
import os
import numpy as np
from PIL import Image
from models.unet import UNet
from models.resnet34_unet import ResNet34_UNet
from oxford_pet import load_dataset
from torch.utils.data import DataLoader
from evaluate import evaluate
from torchvision.transforms import transforms
from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def preprocess_data(img_path):
    data = Image.open(img_path).convert("RGB")
    data = np.array(data.resize((256, 256), Image.BILINEAR))
    data = torch.tensor(data, dtype=torch.float32)
    data /= 255
    data = torch.permute(data, (2, 0, 1))
    return data
        
def to_img(data, mask):
    data = data.squeeze(0).cpu().numpy().transpose((1, 2, 0))
    mask = np.stack((mask,)*3, axis=-1)
    data = data * 255
    mask = mask * 255
    mask = mask.astype('uint8')
    mask = Image.fromarray(mask)
    data = Image.fromarray(data.astype('uint8'))
    return Image.blend(data, mask, alpha=0.5)

def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument("--model", type=str, default="U", help="U: unet / R: resnet_unet")
    parser.add_argument('--data_path', type=str, default="./dataset" ,help='path to the input data')
    parser.add_argument('--batch_size', '-b', type=int, default=1, help='batch size')
    parser.add_argument('--load_model_epoch', '-lme', type=int, default=500, help='load model epoch')
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    if args.model == "U":
        print("Using UNet model")
        model = UNet(3, 1).to(device)
    elif args.model == "R":
        print("Using ResNet34_UNet model")
        model = ResNet34_UNet(3, 1).to(device)
    print(f"Loading model from saved_models/{args.model}/{args.model}_epoch_{args.load_model_epoch}.pth")
    model.load_state_dict(torch.load(f"saved_models/{args.model}/{args.model}_epoch_{args.load_model_epoch}.pth"))
    test_loader = DataLoader(load_dataset(args.data_path, "test"), batch_size=args.batch_size, shuffle=False)
    print("Evaluating model")
    dice_score = evaluate(model, test_loader)
    print(f"Dice Score: {dice_score:.4f}")
    
    list_path = args.data_path + '/annotations/test.txt'
    with open(list_path) as f:
        filenames = f.read().strip('\n').split('\n')
    filenames = [x.split(' ')[0] for x in filenames]
    
    os.makedirs('outputs_imgs', exist_ok=True)
    for file in tqdm(filenames):
        img_path = args.data_path + '/images/' + file + '.jpg'
        data = preprocess_data(img_path)
        data = data.unsqueeze(0).to(device)
        mask = model(data).cpu().detach().numpy().reshape(256, 256)
        mask = mask > 0.5
        new_img = to_img(data, mask)
        new_img.save(f'outputs_imgs/{args.model}/{file}_mask.png')