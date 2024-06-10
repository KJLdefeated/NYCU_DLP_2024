import os
import torch
from torch import nn
import torch.nn.functional as F
from modules.VAE import VQVAE
from modules.UNet import UNet
from modules.DDIM import DDIM
from modules.dataloader import Object
from eval.evaluator import evaluation_model
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import datetime
import json
import argparse
import torchvision
from tqdm import tqdm
import numpy as np

class LDM(object):
    def __init__(
            self,
            batch_size = 32,
            image_size = 64, 
            in_channels = 3, 
            embedd_dim = 3,
            model_channels = 192, 
            out_channels = 3, 
            num_res_blocks = 2, 
            n_heads = 16, 
            transformer_n_layers = 4, 
            attention_resolutions=[2,4,8], 
            dropout=0, 
            channel_mult=[1,2,4,5], 
            num_classes=24,
            n_embd=8192, # for vq model
            context_dim=128, # condition
            ddim_num_steps=50, 
            ddim_discretize="uniform", 
            ddim_eta=0., 
            device='cpu', 
            beta_schedule='cosine',
            vae_ckpt="",
            classifier_guide=1.,
            classifier=None,
        ):
        self.image_size = image_size
        self.batch_size = batch_size
        self.embedd_dim = embedd_dim
        self.vae = VQVAE(n_embed=n_embd, embed_dim=embedd_dim, in_channels=in_channels).to(device).eval()
        self.vae.load_state_dict(torch.load(vae_ckpt, map_location=device))
        self.vae.requires_grad_(False)
        self.latent_size = self.vae.latent_hw
        self.ddim = DDIM(ddim_num_steps=ddim_num_steps, ddim_discretize=ddim_discretize, ddim_eta=ddim_eta, device=device, beta_schedule=beta_schedule, classifier_guide=classifier_guide, classifier=classifier, vae=self.vae)
        self.unet = UNet(image_size, embedd_dim, model_channels, embedd_dim, num_res_blocks, n_heads, transformer_n_layers, attention_resolutions, dropout, channel_mult, num_classes, n_embd, context_dim).to(device).train()

    def train_step(self, x, c):
        t = self.ddim.sample_t((x.size(0),))

        # Encode x with vqvae
        q, _, _ = self.vae.encode(x)
        
        # Forward process
        noise = torch.randn_like(q)
        noise_q = self.ddim.add_noise(q, noise, t)

        # Sampling process
        pred_q = self.unet(noise_q, t, c)

        # Decode
        #pred_x = self.vae.dec_interface(pred_q)
        
        # Compute Loss
        loss = F.mse_loss(pred_q, noise)

        return loss
    
    def infer_step(self, c):
        with torch.no_grad():
            pred = self.ddim.ddim_reverse(model=self.unet, x_size=(self.batch_size, self.embedd_dim, self.latent_size, self.latent_size), condition=c)
            #pred = self.ddim.ddpm_reverse(model=self.unet, x_size=(self.batch_size, self.embedd_dim, self.latent_size, self.latent_size), condition=c)
            pred = self.vae.dec_interface(pred)
        return pred
    
def train(args):
    log_name = f'b{args.batch_size}_lr{args.lr}_nembed{args.n_embd}_embeddim{args.embed_dim}_n_res{args.num_res_blocks}_n_head{args.n_heads}_n_layer{args.transformer_n_layers}_dropout{args.dropout}_ddim{args.ddim_num_steps}_{args.ddim_discretize}_eta{args.ddim_eta}_beta{args.beta_schedule}'
    os.makedirs(f'logs/ldm/{log_name}/tb', exist_ok=True)
    os.makedirs(f'logs/ldm/{log_name}/ckpt', exist_ok=True)
    writer = SummaryWriter(f'logs/ldm/{log_name}/tb')
    train_loader = DataLoader(Object(mode='train'), batch_size=args.batch_size, shuffle=True, num_workers=8, drop_last=True)
    test_loader = [DataLoader(Object(mode='test'), batch_size=args.batch_size, shuffle=False, num_workers=8),
                   DataLoader(Object(mode='new_test'), batch_size=args.batch_size, shuffle=False, num_workers=8)]
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    E = evaluation_model(device)
    model = LDM(
        batch_size=args.batch_size, 
        image_size=args.image_size, 
        in_channels=args.in_channels, 
        embedd_dim=args.embed_dim, 
        model_channels=args.model_channels, 
        out_channels=args.out_channels, 
        num_res_blocks=args.num_res_blocks, 
        n_heads=args.n_heads, 
        transformer_n_layers=args.transformer_n_layers, 
        attention_resolutions=args.attention_resolutions, 
        dropout=args.dropout, 
        channel_mult=args.channel_mult, 
        num_classes=args.num_classes, 
        n_embd=args.n_embd, 
        context_dim=args.context_dim, 
        ddim_num_steps=args.ddim_num_steps, 
        ddim_discretize=args.ddim_discretize, 
        ddim_eta=args.ddim_eta, 
        device=device, 
        beta_schedule=args.beta_schedule, 
        vae_ckpt=args.vae_ckpt,
        classifier_guide=args.classifier_guide,
        classifier=E.resnet18,
        )
    if args.load_ckpt != "":
        model.unet.load_state_dict(torch.load(args.load_ckpt, map_location=device))
    optimizer = torch.optim.Adam(model.unet.parameters(), lr=args.lr)
    for epoch in range(args.epochs):
        losses = []
        model.unet.train()
        print(f'=====Training {epoch}=====')
        for x, c in (pbar := tqdm(train_loader, ncols=140)):
            x, c = x.to(device), c.to(device)
            optimizer.zero_grad()
            loss = model.train_step(x, c)
            loss.backward()
            nn.utils.clip_grad_norm_(model.unet.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())
            pbar.set_description(f'Epoch {epoch}', refresh=False)
            pbar.set_postfix(loss=losses[-1])
            pbar.refresh()
        if epoch % args.per_save == 0 and epoch != 0:
            torch.save(model.unet.state_dict(), f'logs/ldm/{log_name}/ckpt/unet_{epoch}.pt')

        # Test
        scores = [[], []]
        model.unet.eval()
        for i in range(2):
            for label in (pbar := tqdm(test_loader[i], ncols=140)):
                label = label.to(device)
                pred = model.infer_step(label)
                score = E.eval(pred, label)
                scores[i].append(score)
                pbar.set_description(f'Test {i}', refresh=False)
                pbar.set_postfix(score = scores[i][-1])
                pbar.refresh()

        writer.add_scalar('Loss', sum(losses)/len(losses), epoch)
        writer.add_scalar('Test1', sum(scores[0])/len(scores[0]), epoch)
        writer.add_scalar('Test2', sum(scores[1])/len(scores[1]), epoch)
        print(f'Epoch {epoch} | Loss: {sum(losses)/len(losses)} | Test1: {sum(scores[0])/len(scores[0])} | Test2: {sum(scores[1])/len(scores[1])}')

    torch.save(model.unet.state_dict(), f'logs/ldm/{log_name}/ckpt/final.pt')
    writer.close()

def infer(args):
    B = 1
    test_loader = [DataLoader(Object(mode='test'), batch_size=B, shuffle=False, num_workers=8),
                   DataLoader(Object(mode='new_test'), batch_size=B, shuffle=False, num_workers=8)]
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    E = evaluation_model(device)
    model = LDM(
        batch_size=B, 
        image_size=args.image_size, 
        in_channels=args.in_channels, 
        embedd_dim=args.embed_dim, 
        model_channels=args.model_channels, 
        out_channels=args.out_channels, 
        num_res_blocks=args.num_res_blocks, 
        n_heads=args.n_heads, 
        transformer_n_layers=args.transformer_n_layers, 
        attention_resolutions=args.attention_resolutions, 
        dropout=args.dropout, 
        channel_mult=args.channel_mult, 
        num_classes=args.num_classes, 
        n_embd=args.n_embd, 
        context_dim=args.context_dim, 
        ddim_num_steps=args.ddim_num_steps, 
        ddim_discretize=args.ddim_discretize, 
        ddim_eta=args.ddim_eta, 
        device=device, 
        beta_schedule=args.beta_schedule, 
        vae_ckpt=args.vae_ckpt,
        classifier_guide=args.classifier_guide,
        classifier=E.resnet18,
        )
    model.unet.load_state_dict(torch.load(args.load_ckpt, map_location=device))
    model.unet.eval()
    scores = [[], []]
    for i in range(2):
        for j, label in enumerate(pbar := tqdm(test_loader[i], ncols=140)):
            label = label.to(device)
            pred = model.infer_step(label)
            score = E.eval(pred, label)
            # save pred image
            pred = pred * 0.5 + 0.5
            torchvision.utils.save_image(pred, f'img/infer_{i}_{j}_s_{score}.png')
            scores[i].append(score)
            pbar.set_description(f'Test {i}', refresh=False)
            pbar.set_postfix(score = scores[i][-1])
            pbar.refresh()

    print(f'Test1: {sum(scores[0])/len(scores[0])} | Test2: {sum(scores[1])/len(scores[1])}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=3e-6)
    parser.add_argument('--device', type=str, default='1')
    parser.add_argument('--image_size', type=int, default=64)
    parser.add_argument('--in_channels', type=int, default=3)
    parser.add_argument('--embed_dim', type=int, default=6)
    parser.add_argument('--model_channels', type=int, default=192)
    parser.add_argument('--out_channels', type=int, default=3)
    parser.add_argument('--num_res_blocks', type=int, default=2)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--transformer_n_layers', type=int, default=1)
    parser.add_argument('--attention_resolutions', type=list, default=[2,4,8])
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--channel_mult', type=list, default=[1,2,4])
    parser.add_argument('--num_classes', type=int, default=24)
    parser.add_argument('--n_embd', type=int, default=8192)
    parser.add_argument('--context_dim', type=int, default=128)
    parser.add_argument('--ddim_num_steps', type=int, default=50)
    parser.add_argument('--ddim_discretize', type=str, default="uniform")
    parser.add_argument('--ddim_eta', type=float, default=0.)
    parser.add_argument('--beta_schedule', type=str, default='cosine')
    parser.add_argument('--classifier_guide', type=float, default=1.)
    parser.add_argument('--vae_ckpt', type=str, default="logs/vae/b64_lr0.0001_nembed8192_embeddim6/ckpt/300.pt")
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--per_save', type=int, default=10)
    parser.add_argument('--load_model_epoch', type=int, default=-1)
    parser.add_argument('--load_ckpt', type=str, default="")
    parser.add_argument('--infer', action='store_true', default=False)
    args = parser.parse_args()

    if args.infer:
        infer(args)
    else:
        train(args)