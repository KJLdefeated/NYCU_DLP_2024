import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader

from modules import Generator, Gaussian_Predictor, Decoder_Fusion, Label_Encoder, RGB_Encoder

from dataloader import Dataset_Dance
from torchvision.utils import save_image
import random
import torch.optim as optim
from torch import stack

from tqdm import tqdm

import matplotlib.pyplot as plt
from math import log10

from skopt.space import Real, Integer
from skopt import gp_minimize

from torch.utils.tensorboard import SummaryWriter

def Generate_PSNR(imgs1, imgs2, data_range=1.):
    """PSNR for torch tensor"""
    mse = nn.functional.mse_loss(imgs1, imgs2) # wrong computation for batch size > 1
    psnr = 20 * log10(data_range) - 10 * torch.log10(mse)
    return psnr


def kl_criterion(mu, logvar, batch_size):
  KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
  KLD /= batch_size  
  return KLD


class kl_annealing():
    def __init__(self, args, current_epoch=0):
        self.cur_epo = current_epoch
        self.T = args.num_epoch
        self.type = args.kl_anneal_type
        self.cycle = args.kl_anneal_cycle
        self.ratio = args.kl_anneal_ratio
        self.beta = 0
        
    def update(self):
        self.cur_epo += 1
        tau = (self.cur_epo % (self.T // self.cycle)) / (self.T / self.cycle)
        if tau <= self.ratio:
            self.beta = tau / self.ratio
        else:
            self.beta = 1.0
    
    def get_beta(self):
        return self.beta

    def frange_cycle_linear(self, n_iter, start=0.0, stop=1.0,  n_cycle=1, ratio=1):
        L = np.linspace(start, stop, int(n_iter * ratio))
        x = np.arange(0, n_cycle)
        x = np.concatenate([L for _ in range(n_cycle)])
        return x[:n_iter]
        

class VAE_Model(nn.Module):
    def __init__(self, args):
        super(VAE_Model, self).__init__()
        self.args = args
        
        # Modules to transform image from RGB-domain to feature-domain
        self.frame_transformation = RGB_Encoder(3, args.F_dim)
        self.label_transformation = Label_Encoder(3, args.L_dim)
        
        # Conduct Posterior prediction in Encoder
        self.Gaussian_Predictor   = Gaussian_Predictor(args.F_dim + args.L_dim, args.N_dim)
        self.Decoder_Fusion       = Decoder_Fusion(args.F_dim + args.L_dim + args.N_dim, args.D_out_dim)
        
        # Generative model
        self.Generator            = Generator(input_nc=args.D_out_dim, output_nc=3)
        
        self.optim      = optim.Adam(self.parameters(), lr=self.args.lr) if args.optim == "Adam" else optim.AdamW(self.parameters(), lr=self.args.lr)
        
        if args.lr_scheduler == 'MultiStepLR':
            self.scheduler  = optim.lr_scheduler.MultiStepLR(self.optim, milestones=args.lr_milestones, gamma=args.lr_gamma)
        elif args.lr_scheduler == 'ExponentialLR':
            self.scheduler = optim.lr_scheduler.ExponentialLR(self.optim, gamma=args.lr_gamma)
        else:
            raise ValueError("No such scheduler")
        
        self.kl_annealing = kl_annealing(args, current_epoch=0)
        self.mse_criterion = nn.MSELoss()
        self.current_epoch = 0
        
        # Teacher forcing arguments
        self.tfr = args.tfr
        self.tfr_d_step = args.tfr_d_step
        self.tfr_sde = args.tfr_sde
        
        self.train_vi_len = args.train_vi_len
        self.val_vi_len   = args.val_vi_len
        self.batch_size = args.batch_size
        
        self.log_save_path = f"logs/lr_{args.lr}_b_{args.batch_size}_optim_{args.optim}_tfr_{args.tfr}_{args.tfr_sde}_{args.tfr_d_step}_kl_{args.kl_anneal_type}_{args.kl_anneal_cycle}_{args.kl_anneal_ratio}_lr_{args.lr_scheduler}_{args.lr_milestones}_{args.lr_gamma}"
        self.writer = SummaryWriter(self.log_save_path)
        
        
    def forward(self, img, img_last, label):
        frame = self.frame_transformation(img)
        frame_last = self.frame_transformation(img_last)
        pose = self.label_transformation(label)
        z, mu, logvar = self.Gaussian_Predictor(frame, pose)
        kl_loss = kl_criterion(mu, logvar, self.batch_size)
        fusion = self.Decoder_Fusion(frame_last, pose, z)
        pred = self.Generator(fusion)
        return pred, kl_loss
        
    def training_stage(self):
        losses = []
        val_losses = []
        psnrs = []
        for i in range(self.args.num_epoch):
            train_loader = self.train_dataloader()
            adapt_TeacherForcing = True if random.random() < self.tfr else False
            
            ep_loss = 0
            
            for (img, label) in (pbar := tqdm(train_loader, ncols=120)):
                img = img.to(self.args.device)
                label = label.to(self.args.device)
                loss = self.training_one_step(img, label, adapt_TeacherForcing)
                ep_loss += loss.cpu()
                
                beta = self.kl_annealing.get_beta()
                if adapt_TeacherForcing:
                    self.tqdm_bar('train [TeacherForcing: ON, {:.1f}], beta: {}'.format(self.tfr, beta), pbar, loss.detach().cpu(), lr=self.scheduler.get_last_lr()[0])
                else:
                    self.tqdm_bar('train [TeacherForcing: OFF, {:.1f}], beta: {}'.format(self.tfr, beta), pbar, loss.detach().cpu(), lr=self.scheduler.get_last_lr()[0])
                
            val_loss, psnr = self.eval()

            if self.current_epoch % self.args.per_save == 0 or psnr > 30.0:
                os.makedirs(f"{self.log_save_path}/ckpt", exist_ok=True)
                self.save(f"{self.log_save_path}/ckpt/{self.current_epoch}.ckpt")
            
            if torch.isnan(psnr):
                print("PSNR is nan")
                return 0
            
            self.writer.add_scalar('train loss', ep_loss / len(train_loader), self.current_epoch)
            self.writer.add_scalar('val loss', val_loss, self.current_epoch)
            self.writer.add_scalar('val PSNR', psnr, self.current_epoch)
            self.writer.add_scalar('lr', self.scheduler.get_last_lr()[0], self.current_epoch)
            self.writer.add_scalar('beta', self.kl_annealing.get_beta(), self.current_epoch)
            self.writer.add_scalar('tfr', self.tfr, self.current_epoch)
            
            losses.append(ep_loss / len(train_loader))
            val_losses.append(val_loss.numpy())
            psnrs.append(psnr.numpy())
            
            self.current_epoch += 1
            self.scheduler.step()
            self.teacher_forcing_ratio_update()
            self.kl_annealing.update()
            
        np.save(self.log_save_path + "/losses.npy", np.array(losses))
        np.save(self.log_save_path + "/val_losses.npy", np.array(val_losses))
        np.save(self.log_save_path + "/psnrs.npy", np.array(psnrs))
        
        return max(psnrs[10:])
            
            
    @torch.no_grad()
    def eval(self):
        val_loader = self.val_dataloader()
        for (img, label) in (pbar := tqdm(val_loader, ncols=120)):
            img = img.to(self.args.device)
            label = label.to(self.args.device)
            loss, psnr = self.val_one_step(img, label)
            self.tqdm_bar(f'val | PSNR: {psnr:.2f}', pbar, loss.detach().cpu(), lr=self.scheduler.get_last_lr()[0])
        return loss.cpu().detach(), psnr.cpu().detach()
    
    def training_one_step(self, img, label, adapt_TeacherForcing):
        img_last = img[:, 0]
        KL = 0
        MSE = 0
        for i in range(self.train_vi_len-1):
            img_in = img[:, i+1]
            label_in = label[:, i+1]
            if adapt_TeacherForcing:
                pred, kl_loss = self(img_in, img[:, i], label_in)
            else:
                pred, kl_loss = self(img_in, img_last, label_in)
            KL += kl_loss
            MSE += self.mse_criterion(pred, img_in)
            img_last = pred.detach()
        self.optim.zero_grad()
        loss = MSE + KL * self.kl_annealing.get_beta()
        loss.backward()
        self.optimizer_step()
        return loss.detach()
    
    def val_one_step(self, img, label):
        with torch.no_grad():
            img_last = img[:, 0]
            KL = 0
            MSE = 0
            PSNR = 0
            for i in range(self.val_vi_len-1):
                img_in = img[:, i+1]
                label_in = label[:, i+1]
                pred, kl_loss = self(img_in, img_last, label_in)
                KL += kl_loss
                MSE += self.mse_criterion(pred, img_in)
                PSNR += Generate_PSNR(pred, img_in)
                img_last = pred.detach()
            PSNR /= self.val_vi_len
            loss = MSE + KL * self.kl_annealing.get_beta()
        return loss.detach(), PSNR.detach()
                
    def make_gif(self, images_list, img_name):
        new_list = []
        for img in images_list:
            new_list.append(transforms.ToPILImage()(img))
            
        new_list[0].save(img_name, format="GIF", append_images=new_list,
                    save_all=True, duration=40, loop=0)
    
    def train_dataloader(self):
        transform = transforms.Compose([
            transforms.Resize((self.args.frame_H, self.args.frame_W)),
            transforms.ToTensor()
        ])

        dataset = Dataset_Dance(root=self.args.DR, transform=transform, mode='train', video_len=self.train_vi_len, \
                                                partial=args.fast_partial if self.args.fast_train else args.partial)
        if self.current_epoch > self.args.fast_train_epoch:
            self.args.fast_train = False
            
        train_loader = DataLoader(dataset,
                                  batch_size=self.batch_size,
                                  num_workers=self.args.num_workers,
                                  drop_last=True,
                                  shuffle=False)  
        return train_loader
    
    def val_dataloader(self):
        transform = transforms.Compose([
            transforms.Resize((self.args.frame_H, self.args.frame_W)),
            transforms.ToTensor()
        ])
        dataset = Dataset_Dance(root=self.args.DR, transform=transform, mode='val', video_len=self.val_vi_len, partial=1.0)  
        val_loader = DataLoader(dataset,
                                  batch_size=1,
                                  num_workers=self.args.num_workers,
                                  drop_last=True,
                                  shuffle=False)  
        return val_loader
    
    def teacher_forcing_ratio_update(self):
        if self.current_epoch > self.tfr_sde:
            self.tfr -= self.tfr_d_step
            self.tfr = max(0.0, self.tfr)
            
    def tqdm_bar(self, mode, pbar, loss, lr):
        pbar.set_description(f"({mode}) Epoch {self.current_epoch}, lr:{lr}" , refresh=False)
        pbar.set_postfix(loss=float(loss), refresh=False)
        pbar.refresh()
        
    def save(self, path):
        torch.save({
            "state_dict": self.state_dict(),
            "optimizer": self.state_dict(),  
            "lr"        : self.scheduler.get_last_lr()[0],
            "tfr"       :   self.tfr,
            "last_epoch": self.current_epoch
        }, path)
        print(f"save ckpt to {path}")

    def load_checkpoint(self):
        if self.args.ckpt_path != None:
            checkpoint = torch.load(self.args.ckpt_path)
            self.load_state_dict(checkpoint['state_dict'], strict=True) 
            self.args.lr = checkpoint['lr']
            self.tfr = checkpoint['tfr']
            
            self.optim      = optim.Adam(self.parameters(), lr=self.args.lr)
            self.scheduler  = optim.lr_scheduler.MultiStepLR(self.optim, milestones=[2, 4], gamma=0.1)
            self.kl_annealing = kl_annealing(self.args, current_epoch=checkpoint['last_epoch'])
            self.current_epoch = checkpoint['last_epoch']

    def optimizer_step(self):
        nn.utils.clip_grad_norm_(self.parameters(), 1.)
        self.optim.step()

def main(args):
    model = VAE_Model(args).to(args.device)
    model.load_checkpoint()
    if args.test:
        model.eval()
    else:
        PSNR = model.training_stage()
    return PSNR
    
search_space = [    
    Real(0, 0.001,name='lr'),
    Integer(5, 50,name='tfr_sde'),
    Real(0.01, 0.5,name='tfr_d_step'),
    Integer(5, 20,name='kl_anneal_cycle'),
    Real(0.01, 0.5,name='kl_anneal_ratio'),
]

def objective(params):
    print(params)
    args.lr = params[0]
    args.tfr_sde = params[1]
    args.tfr_d_step = params[2]
    args.kl_anneal_cycle = params[3]
    args.kl_anneal_ratio = params[4]
    return -main(args)
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--batch_size',    type=int,    default=2)
    parser.add_argument('--lr',            type=float,  default=0.001,     help="initial learning rate")
    parser.add_argument('--device',        type=str, choices=["cuda", "cpu"], default="cuda")
    parser.add_argument('--optim',         type=str, choices=["Adam", "AdamW"], default="Adam")
    parser.add_argument('--gpu',           type=int, default=1)
    parser.add_argument('--test',          action='store_true')
    parser.add_argument('--store_visualization',      action='store_true', help="If you want to see the result while training")
    parser.add_argument('--DR',            type=str, default="Dataset", help="Your Dataset Path")
    parser.add_argument('--save_root',     type=str, default="ckpt", help="The path to save your data")
    parser.add_argument('--num_workers',   type=int, default=4)
    parser.add_argument('--num_epoch',     type=int, default=200,     help="number of total epoch")
    parser.add_argument('--per_save',      type=int, default=1,      help="Save checkpoint every seted epoch")
    parser.add_argument('--partial',       type=float, default=1.0,  help="Part of the training dataset to be trained")
    parser.add_argument('--train_vi_len',  type=int, default=16,     help="Training video length")
    parser.add_argument('--val_vi_len',    type=int, default=630,    help="valdation video length")
    parser.add_argument('--frame_H',       type=int, default=32,     help="Height input image to be resize")
    parser.add_argument('--frame_W',       type=int, default=64,     help="Width input image to be resize")
    
    
    # Module parameters setting
    parser.add_argument('--F_dim',         type=int, default=128,    help="Dimension of feature human frame")
    parser.add_argument('--L_dim',         type=int, default=32,     help="Dimension of feature label frame")
    parser.add_argument('--N_dim',         type=int, default=12,     help="Dimension of the Noise")
    parser.add_argument('--D_out_dim',     type=int, default=192,    help="Dimension of the output in Decoder_Fusion")
    
    # Teacher Forcing strategy
    parser.add_argument('--tfr',           type=float, default=1.0,  help="The initial teacher forcing ratio")
    parser.add_argument('--tfr_sde',       type=int,   default=10,   help="The epoch that teacher forcing ratio start to decay")
    parser.add_argument('--tfr_d_step',    type=float, default=0.1,  help="Decay step that teacher forcing ratio adopted")
    parser.add_argument('--ckpt_path',     type=str,    default=None,help="The path of your checkpoints")   
    
    # Training Strategy
    parser.add_argument('--fast_train',         action='store_true')
    parser.add_argument('--fast_partial',       type=float, default=0.4,    help="Use part of the training data to fasten the convergence")
    parser.add_argument('--fast_train_epoch',   type=int, default=5,        help="Number of epoch to use fast train mode")
    
    # Kl annealing stratedy arguments
    parser.add_argument('--kl_anneal_type',     type=str, default='Cyclical',       help="")
    parser.add_argument('--kl_anneal_cycle',    type=int, default=20,               help="")
    parser.add_argument('--kl_anneal_ratio',    type=float, default=1,              help="")
    
    # LR scheduler
    parser.add_argument('--lr_scheduler',       type=str, default='MultiStepLR',    help="")
    parser.add_argument('--lr_milestones',      type=list, default=[2, 10],           help="")
    parser.add_argument('--lr_gamma',           type=float, default=0.1,            help="")

    args = parser.parse_args()
    main(args)
    # result = gp_minimize(objective, search_space, n_calls=15, random_state=0)

    # print("Best hyperparameters: ", result.x)
    # print("Best objective value: ", result.fun)
    # print("Hyperparameters tried: ", result.x_iters)
    # print("Objective values at each step: ", result.func_vals)