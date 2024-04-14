import pandas as pd
from PIL import Image
from torch.utils import data
import torch
import torchvision.transforms as transforms

def getData(mode):
    if mode == 'train':
        df = pd.read_csv('dataset/train.csv')
        path = df['filepaths'].tolist()
        label = df['label_id'].tolist()
        return path, label
    elif mode == 'test':
        df = pd.read_csv('dataset/test.csv')
        path = df['filepaths'].tolist()
        label = df['label_id'].tolist()
        return path, label
    elif mode == 'valid':
        df = pd.read_csv('dataset/valid.csv')
        path = df['filepaths'].tolist()
        label = df['label_id'].tolist()
        return path, label

class BufferflyMothLoader(data.Dataset):
    def __init__(self, root, mode):
        super(BufferflyMothLoader, self).__init__()
        """
        Argsś
            mode : Indicate procedure status(training or testing)

            self.img_name (string list): String list that store all image names.
            self.label (int or float list): Numerical list that store all ground truth label values.
        """
        self.root = root
        self.img_name, self.label = getData(mode)
        self.mode = mode
        self.images = []
        self.labels = []
        for index in range(len(self.img_name)):
            img = Image.open(self.root + '/' + self.img_name[index])
            img = img.convert('RGB')
            label = self.label[index]
            self.images.append(img)
            self.labels.append(label)
        print("> Found %d images..." % (len(self.img_name)))  

    def __len__(self):
        """'return the size of dataset"""
        return len(self.img_name)

    def __getitem__(self, index):
        """something you should implement here"""

        """
           step1. Get the image path from 'self.img_name' and load it.
                  hint : path = root + self.img_name[index] + '.jpg'
           
           step2. Get the ground truth label from self.label
                     
           step3. Transform the .jpg rgb images during the training phase, such as resizing, random flipping, 
                  rotation, cropping, normalization etc. But at the beginning, I suggest you follow the hints. 
                       
                  In the testing phase, if you have a normalization process during the training phase, you only need 
                  to normalize the data. 
                  
                  hints : Convert the pixel value to [0, 1]
                          Transpose the image shape from [H, W, C] to [C, H, W]
                         
            step4. Return processed image and label
        """

        img = self.images[index]
        label = self.labels[index]

        if self.mode == 'train':
            # Data augmentation
            transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomApply(torch.nn.ModuleList([transforms.ColorJitter(brightness=.5, hue=.3)]), p=0.25),
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
            ])
        else:
            # Data normalization
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
            ])
        
        img = transform(img)
        torch.permute(img, (2, 0, 1))
        for i in range(3):
            img[i] = (img[i] - img[i].min()) / (img[i].max() - img[i].min())
        
        label = torch.tensor(label)
        return img, label
