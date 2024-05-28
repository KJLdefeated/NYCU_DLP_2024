import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms
import numpy as np
from torchvision.datasets.folder import default_loader as imgloader
import json
import glob

class Object(Dataset):
    def __init__(self, mode='train'):
        super(Object).__init__()
        self.mode = mode
        assert mode in ['train', 'test', 'new_test']
        with open('eval/objects.json', 'r') as f:
            self.objects = json.load(f)
        with open(f'eval/{mode}.json', 'r') as f:
            self.labels = json.load(f)
        if mode == 'train':
            self.images = glob.glob('iclevr/*.png')
            self.images.sort(key=lambda x: int(x.split('/')[-1].split('.')[0].split('_')[-1]) + 10 * int(x.split('/')[-1].split('.')[0].split('_')[-2]))
            self.images = self.images[:-3]
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        if self.mode == 'train':
            image = self.transform(imgloader(self.images[idx]))
            label_name = self.labels[self.images[idx].split('/')[-1]]
            label = [0] * 24
            for i in label_name: label[self.objects[i]] = 1
            label = torch.tensor(np.array(label), dtype=torch.float32)
            return image, label
        else:
            label = [0] * 24
            for i in self.labels[idx]: label[self.objects[i]] = 1
            label = torch.tensor(np.array(label), dtype=torch.float32)
            return label