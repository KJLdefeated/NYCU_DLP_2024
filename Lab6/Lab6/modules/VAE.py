import torch
from torch import nn
import os
import torch.nn.functional as F
from taming.modules.vqvae.quantize import VectorQuantizer
from modules.layers import MyConvo2d, ResidualBlock
from torch.utils.data import DataLoader
from modules.dataloader import Object
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torchvision
from taming.modules.discriminator.model import NLayerDiscriminator, weights_init
from taming.modules.losses.lpips import LPIPS
from taming.modules.losses.vqperceptual import hinge_d_loss, vanilla_d_loss

class Encoder(nn.Module):
    def __init__(self, in_channels = 3, ch = 128, ch_mult=[1, 2, 4], z_dim = 3, hw = 64):
        super(Encoder, self).__init__()
        self.in_channels = in_channels
        self.conv1 = MyConvo2d(in_channels, ch, 3)
        self.blocks = []
        block_in = ch
        for i in range(len(ch_mult)):
            block_out = ch*ch_mult[i]
            self.blocks.append(ResidualBlock(block_in, block_out, 3, resample = 'down', hw=hw))
            block_in = block_out
            hw = hw // 2
        self.blocks.append(ResidualBlock(block_out, block_out, 3, resample = None))
        self.blocks = nn.Sequential(*self.blocks)
        self.out = MyConvo2d(block_out, z_dim, 3)

    def forward(self, input):
        output = self.conv1(input)
        output = self.blocks(output)
        output = self.out(output)
        return output
    
class Decoder(nn.Module):
    def __init__(self, out_channels = 3, ch = 128, ch_mult=[1, 2, 4], z_dim = 3):
        super(Decoder, self).__init__()
        self.out_channels = out_channels
        block_in = ch*ch_mult[-1]
        self.conv1 = MyConvo2d(z_dim, block_in, 3)
        self.blocks = []
        self.blocks.append(ResidualBlock(block_in, block_in, 3, resample = None))
        for i in range(len(ch_mult)-1, -1, -1):
            block_out = ch*ch_mult[i]
            self.blocks.append(ResidualBlock(block_in, block_out, 3, resample = 'up'))
            block_in = block_out
        self.blocks = nn.Sequential(*self.blocks)
        self.out = MyConvo2d(block_in, out_channels, 3)

    def forward(self, input):
        output = self.conv1(input)
        output = self.blocks(output)
        output = self.out(output)
        return output
    
class VQVAE(nn.Module):
    def __init__(self, n_embed = 8192, embed_dim = 3, in_channels = 3):
        super(VQVAE, self).__init__()
        self.n_embed = n_embed
        self.embed_dim = embed_dim
        ch_mult = [1, 2, 4]
        self.latent_hw = 64 // 2**len(ch_mult)
        self.encoder = Encoder(in_channels=in_channels, z_dim=embed_dim, ch = 128, ch_mult=ch_mult, hw = 64)
        self.decoder = Decoder(z_dim=embed_dim, out_channels=in_channels)
        self.quantize = VectorQuantizer(n_embed, embed_dim, 0.25)
        self.discriminator = NLayerDiscriminator(input_nc=in_channels,n_layers=3,use_actnorm=False,ndf=64).apply(weights_init)
        self.perceptual_loss = LPIPS().eval()
        self.discriminator_weight = 0.75

    def enc_interface(self, input):
        return self.encoder(input)
    
    def dec_interface(self, input):
        quant, _, _ = self.quantize(input)
        return self.decoder(quant)

    def encode(self, input):
        h = self.encoder(input)
        quant, emb_loss, info = self.quantize(h)
        return quant, emb_loss, info
    
    def decode(self, quant):
        return self.decoder(quant)
    
    def forward(self, input):
        quant, diff, (_,_,ind) = self.encode(input)
        dec = self.decode(quant)
        return dec, diff, ind
    
    def train_step(self, input, opt_idx):
        dec, emb_loss, _ = self(input)
        recon_loss = F.mse_loss(dec, input) + self.perceptual_loss(dec, input).mean()
        if opt_idx == 0:
            logits_fake = self.discriminator(dec.contiguous())
            g_loss = -torch.mean(logits_fake)
            d_weight = self.calculate_adaptive_weight(recon_loss, g_loss, last_layer=self.decoder.out.conv.weight)
            loss = recon_loss + d_weight * g_loss + emb_loss
            return loss
        else:
            logits_fake = self.discriminator(dec.contiguous())
            logits_real = self.discriminator(input.contiguous())
            d_loss = hinge_d_loss(logits_real, logits_fake)
            return d_loss
        
    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
        g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * self.discriminator_weight
        return d_weight

def train(args):
    log_name = f'b{args.batch_size}_lr{args.lr}_nembed{args.n_embed}_embeddim{args.embed_dim}'
    os.makedirs(f'logs/vae/{log_name}/tb', exist_ok=True)
    os.makedirs(f'logs/vae/{log_name}/ckpt', exist_ok=True)
    writer = SummaryWriter(f'logs/vae/{log_name}/tb')
    train_loader = DataLoader(Object(mode='train'), batch_size=args.batch_size, shuffle=True, num_workers=8, drop_last=True)
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    model = VQVAE(n_embed=args.n_embed, embed_dim=args.embed_dim).to(device)
    optimizer = [
        torch.optim.Adam(list(model.encoder.parameters())+
                        list(model.decoder.parameters())+
                        list(model.quantize.parameters()), lr=args.lr, betas=(0.5, 0.9)), # ae optimizer
        torch.optim.Adam(model.discriminator.parameters(), lr=args.lr, betas=(0.5, 0.9))   # discriminator optimizer  
    ]

    for epoch in range(args.epochs):
        losses = []
        if (epoch + 1) % args.disc_freq == 0:
            opt_idx = 0 # ae
            mode = 'ae'
        else:
            opt_idx = 1
            mode = 'disc'
        for image, label in (pbar:=tqdm(train_loader, ncols=140)):
            image = image.to(device)
            optimizer[opt_idx].zero_grad()
            loss = model.train_step(image, opt_idx)
            loss.backward()
            optimizer[opt_idx].step()
            losses.append(loss.item())
            pbar.set_description(f'Epoch {epoch}', refresh=False)
            pbar.set_postfix(loss = loss.item())
            pbar.refresh()
        writer.add_scalar('Loss', sum(losses)/len(losses), epoch)
        print(f'Epoch {epoch} {mode} Loss: {sum(losses)/len(losses)}')
        if epoch % args.per_save == 0:
            torch.save(model.state_dict(), f'logs/vae/{log_name}/ckpt/{epoch}.pt')

def infer(args):
    log_name = f'b{args.batch_size}_lr{args.lr}_nembed{args.n_embed}_embeddim{args.embed_dim}'
    os.makedirs(f'logs/vae/{log_name}/infer', exist_ok=True)
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    train_loader = DataLoader(Object(mode='train'), batch_size=args.batch_size, shuffle=False, num_workers=8, drop_last=True)
    model = VQVAE(n_embed=args.n_embed, embed_dim=args.embed_dim).to(device)
    model.load_state_dict(torch.load(f'logs/vae/{log_name}/ckpt/{args.load_model_epoch}.pt'))
    model.eval()
    val_losses = []
    for image, _ in (pbar:=tqdm(train_loader, ncols=140)):
        image = image.to(device)
        rec, _, _ = model(image)
        val_loss = F.mse_loss(rec, image)
        val_losses.append(val_loss.item())
        #rec = rec[0].detach().cpu()
        #rec = (rec + 1) / 2
        #torchvision.utils.save_image(rec, f'logs/vae/{log_name}/infer/{pbar.n}.png')
        pbar.set_postfix(loss = val_loss.item())
    print(f'Validation Loss: {sum(val_losses)/len(val_losses)}')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--n_embed', type=int, default=8192)
    parser.add_argument('--embed_dim', type=int, default=6)
    parser.add_argument('--disc_freq', type=int, default=2)
    parser.add_argument('--device', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--per_save', type=int, default=10)
    parser.add_argument('--load_model_epoch', type=int, default=490)
    parser.add_argument('--infer', action='store_true')
    args = parser.parse_args()
    if not args.infer:
        train(args)
    else:
        infer(args)