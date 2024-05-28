import torch
import os
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch import autograd
from modules.layers import MyConvo2d, ResidualBlock
from modules.dataloader import Object
from eval.evaluator import evaluation_model
from tqdm import tqdm
import argparse

class Generator(nn.Module):
    def __init__(self, dim=64):
        super(Generator, self).__init__()
        self.dim = dim
        self.ln1 = nn.Linear(self.dim + 24, self.dim * 4 * 4 * 8)
        self.rb1 = ResidualBlock(8*self.dim, 8*self.dim, 3, resample = 'up')
        self.rb2 = ResidualBlock(8*self.dim, 4*self.dim, 3, resample = 'up')
        self.rb3 = ResidualBlock(4*self.dim, 2*self.dim, 3, resample = 'up')
        self.rb4 = ResidualBlock(2*self.dim, 1*self.dim, 3, resample = 'up')
        self.bn  = nn.BatchNorm2d(self.dim)

        self.conv1 = MyConvo2d(1*self.dim, 3, 3)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        
    def forward(self, noise, label):
        input = torch.cat((noise, label), 1)
        output = self.ln1(input)
        output = output.view(-1, 8*self.dim, 4, 4)
        output = self.rb1(output)
        output = self.rb2(output)
        output = self.rb3(output)
        output = self.rb4(output)
        output = self.bn(output)
        output = self.relu(output)
        output = self.conv1(output)
        output = self.tanh(output)
        return output
    
class Discriminator(nn.Module):
    def __init__(self, dim=64):
        super(Discriminator, self).__init__()
        self.dim = dim
        self.conv1 = MyConvo2d(3, self.dim, 3)
        self.rb1 = ResidualBlock(self.dim, 2*self.dim, 3, resample = 'down', hw=self.dim)
        self.rb2 = ResidualBlock(2*self.dim, 4*self.dim, 3, resample = 'down', hw=int(self.dim/2))
        self.rb3 = ResidualBlock(4*self.dim, 8*self.dim, 3, resample = 'down', hw=int(self.dim/4))
        self.rb4 = ResidualBlock(8*self.dim, 8*self.dim, 3, resample = 'down', hw=int(self.dim/8))
        self.ln1 = nn.Linear(8*self.dim * 4 * 4, 1)
        self.lab_enc = nn.Linear(24, 8*self.dim * 4 * 4)
        self.relu = nn.ReLU()
        
    def forward(self, input, label):
        output = self.conv1(input.contiguous())
        output = self.rb1(output)
        output = self.rb2(output)
        output = self.rb3(output)
        output = self.rb4(output)
        output = self.relu(output)
        h = output.view(-1, 8*self.dim * 4 * 4)
        output = self.ln1(h)
        label = self.lab_enc(label)
        output += (h * label).sum(1, keepdim=True)
        return output.view(-1)

def calc_gradient_penalty(netD, real_data, fake_data, labels, batch_size, dim, device, gp_lambda):
    alpha = torch.rand(batch_size, 1)
    alpha = alpha.expand(batch_size, int(real_data.nelement()/batch_size)).contiguous()
    alpha = alpha.view(batch_size, 3, dim, dim)
    alpha = alpha.to(device)
    
    fake_data = fake_data.view(batch_size, 3, dim, dim)
    interpolates = alpha * real_data.detach() + ((1 - alpha) * fake_data.detach())

    interpolates = interpolates.to(device)
    interpolates.requires_grad_(True)

    disc_interpolates = netD(interpolates, labels)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradients = gradients.view(gradients.size(0), -1)                              
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * gp_lambda
    return gradient_penalty

def train(args):
    log_name = f'b{args.batch_size}_lr{args.lr}_optim{args.optim}_dim{args.dim}_gp{args.gp}_ngen{args.gen_iter}_ndis{args.dis_iter}'
    os.makedirs(f'logs/wgan/{log_name}/tb', exist_ok=True)
    os.makedirs(f'logs/wgan/{log_name}/ckpt', exist_ok=True)
    writer = SummaryWriter(f'logs/wgan/{log_name}/tb')
    train_loader = DataLoader(Object(mode='train'), batch_size=args.batch_size, shuffle=True, num_workers=8, drop_last=True)
    test_loader = [DataLoader(Object(mode='test'), batch_size=32, shuffle=False, num_workers=8),
                   DataLoader(Object(mode='new_test'), batch_size=32, shuffle=False, num_workers=8)]
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    G = Generator(args.dim).to(device)
    D = Discriminator(args.dim).to(device)
    E = evaluation_model(device)
    if args.optim == 'adam':
        G_opt = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(0.5, 0.999))
        D_opt = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(0.5, 0.999))
    elif args.optim == 'rmsprop':
        G_opt = torch.optim.RMSprop(G.parameters(), lr=args.lr)
        D_opt = torch.optim.RMSprop(D.parameters(), lr=args.lr)
    else:
        raise NotImplementedError
    
    for epoch in range(args.epochs):
        D_losses = []
        G_losses = []
        W_dists = []
        print(f'=====Training {epoch}=====')
        for (image, label) in (pbar := tqdm(train_loader, ncols=140)):
            image, label = image.to(device), label.to(device)

            # Train Discriminator
            tot_d_loss = 0
            tot_g_loss = 0
            tot_w = 0
            D.train()
            G.eval()
            for p in D.parameters(): p.requires_grad_(True)
            for _ in range(args.dis_iter):
                noise = torch.randn(args.batch_size, args.dim).to(device)
                with torch.no_grad():
                    fake_image = G(noise, label)
                real = D(image, label).mean()
                fake = D(fake_image, label).mean()
                gp = calc_gradient_penalty(D, image, fake_image, label, args.batch_size, args.dim, device, args.gp)
                D_loss = fake - real + gp
                W_dist = fake - real
                D_opt.zero_grad()
                D_loss.backward()
                D_opt.step()
                tot_d_loss += D_loss.item()
                tot_w += W_dist.item()

            # Train Generator
            D.eval()
            G.train()
            for p in D.parameters(): p.requires_grad_(False)
            for _ in range(args.gen_iter):
                noise = torch.randn(args.batch_size, 64).to(device)
                fake_image = G(noise, label)
                fake = D(fake_image, label).mean()
                G_loss = -fake
                G_opt.zero_grad()
                G_loss.backward()
                G_opt.step()
                tot_g_loss += G_loss.item()
            
            D_losses.append(tot_d_loss/args.dis_iter)
            G_losses.append(tot_g_loss/args.gen_iter)
            W_dists.append(tot_w/args.dis_iter)
            pbar.set_description(f'Epoch {epoch}', refresh=False)
            pbar.set_postfix(D_loss=D_losses[-1], G_loss=G_losses[-1], W_dist=W_dists[-1])
            pbar.refresh()
        
        G.eval()
        scores = []
        for i in range(2):
            for label in (pbar := tqdm(test_loader[i], ncols=140)):
                label = label.to(device)
                noise = torch.randn(32, args.dim).to(device)
                with torch.no_grad():
                    fake_image = G(noise, label)
                score = E.eval(fake_image, label)
                scores.append(score)
                pbar.set_description(f'Test {i}', refresh=False)
                pbar.set_postfix(score=score)
                pbar.refresh()

        writer.add_scalar('Discriminator Loss', sum(D_losses)/len(D_losses), epoch)
        writer.add_scalar('Generator Loss', sum(G_losses)/len(G_losses), epoch)
        writer.add_scalar('Wasserstein Distance', sum(W_dists)/len(W_dists), epoch)
        writer.add_scalar('Test1', scores[0], epoch)
        writer.add_scalar('Test2', scores[1], epoch)
        print(f'Epoch {epoch} | G Loss: {sum(G_losses)/len(G_losses):.4f} | D Loss: {sum(D_losses)/len(D_losses):.4f} | W Dist: {sum(W_dists)/len(W_dists):.4f} | Test1: {scores[0]:.4f} | Test2: {scores[1]:.4f}')

        if epoch % args.per_save == 0:
            torch.save(G.state_dict(), f'logs/wgan/{log_name}/ckpt/G_{epoch}.pth')
            torch.save(D.state_dict(), f'logs/wgan/{log_name}/ckpt/D_{epoch}.pth')

    writer.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--optim', type=str, default='adam')
    parser.add_argument('--dim', type=int, default=64)
    parser.add_argument('--gp', type=float, default=10)
    parser.add_argument('--gen_iter', type=int, default=1)
    parser.add_argument('--dis_iter', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--per_save', type=int, default=10)
    args = parser.parse_args()
    train(args)