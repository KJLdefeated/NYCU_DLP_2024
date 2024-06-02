import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.functional as F

def make_beta_schedule(schedule="linear", num_timesteps=1000, start=1e-5, end=1e-2):
    if schedule == "linear":
        betas = torch.linspace(start, end, num_timesteps)
    elif schedule == "sigmoid":
        betas = torch.linspace(-6, 6, num_timesteps)
        betas = torch.sigmoid(betas) * (end - start) + start
    elif schedule == "cosine" or schedule == "cosine_reverse":
        max_beta = 0.999
        cosine_s = 0.008
        betas = torch.tensor(
            [min(1 - (math.cos(((i + 1) / num_timesteps + cosine_s) / (1 + cosine_s) * math.pi / 2) ** 2) / (
                    math.cos((i / num_timesteps + cosine_s) / (1 + cosine_s) * math.pi / 2) ** 2), max_beta) for i in
             range(num_timesteps)])
    elif schedule == "cosine_anneal":
        betas = torch.tensor(
            [start + 0.5 * (end - start) * (1 - math.cos(t / (num_timesteps - 1) * math.pi)) for t in
             range(num_timesteps)])
    return betas

class DDIM(object):
    def __init__(self, ddim_num_steps=50, ddim_discretize="uniform", ddim_eta=0., device='cpu', beta_schedule='cosine'):
        self.device = device
        self.num_timesteps = 1000 # DDPM num timesteps
        self.ddim_num_steps = ddim_num_steps
        self.ddim_discretize = ddim_discretize
        self.eta = ddim_eta
        self.ddim_timesteps = self.make_timesteps(discr_method=ddim_discretize, num_ddim_timesteps=ddim_num_steps, num_ddpm_timesteps=self.num_timesteps)

        # DDPM parameters
        self.beta_schedule = beta_schedule
        betas = make_beta_schedule(schedule=beta_schedule, num_timesteps=self.num_timesteps, start=0.0001, end=0.02)
        betas = self.betas = betas.float().to(self.device)
        self.betas_sqrt = torch.sqrt(betas)
        alphas = 1.0 - betas
        self.alphas = alphas
        self.one_minus_betas_sqrt = torch.sqrt(alphas)
        self.alphas_cumprod = alphas.cumprod(dim=0)
        self.alphas_bar_sqrt = torch.sqrt(self.alphas_cumprod)
        self.one_minus_alphas_bar_sqrt = torch.sqrt(1 - self.alphas_cumprod)
        alphas_cumprod_prev = torch.cat([torch.ones(1).to(self.device), self.alphas_cumprod[:-1]], dim=0)
        self.alphas_cumprod_prev = alphas_cumprod_prev
        self.posterior_mean_coeff_1 = (betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - self.alphas_cumprod))
        self.posterior_mean_coeff_2 = (torch.sqrt(alphas) * (1 - alphas_cumprod_prev) / (1 - self.alphas_cumprod))
        posterior_variance = (betas * (1.0 - alphas_cumprod_prev) / (1.0 - self.alphas_cumprod))
        self.posterior_variance = posterior_variance
        self.logvar = betas.log()

        # DDIM parameters
        self.ddim_alphas = self.alphas_cumprod[self.ddim_timesteps]
        self.ddim_alphas_prev = torch.tensor([self.alphas_cumprod[0]] + self.alphas_cumprod[self.ddim_timesteps[:-1]].tolist()).to(device)
        self.ddim_sigmas = self.eta * torch.sqrt((1 - self.ddim_alphas_prev) / (1 - self.ddim_alphas) * (1 - self.ddim_alphas / self.ddim_alphas_prev))
        self.ddim_sqrt_one_minus_alphas = torch.sqrt(1. - self.ddim_alphas)
        self.sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt((1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) * (1 - self.alphas_cumprod / self.alphas_cumprod_prev)) 
        
    def sample_t(self, size=(1,)):
       """Samples batches of time steps to use."""
       t_max = int(self.num_timesteps) - 1
       t = torch.randint(high=t_max, size=size, device=self.device)
       return t

    def make_timesteps(self, discr_method, num_ddim_timesteps, num_ddpm_timesteps):
        if discr_method == 'uniform':
            c = num_ddpm_timesteps // num_ddim_timesteps
            ddim_timesteps = np.asarray(list(range(0, num_ddpm_timesteps, c)))
        elif discr_method == 'quad':
            ddim_timesteps = ((np.linspace(0, np.sqrt(num_ddpm_timesteps * .8), num_ddim_timesteps)) ** 2).astype(int)
        else:
            raise NotImplementedError(f'There is no ddim discretization method called "{discr_method}"')

        assert ddim_timesteps.shape[0] == num_ddim_timesteps
        # add one to get the final alpha values right (the ones from first scale to data during sampling)
        steps_out = ddim_timesteps + 1
        # print(f'Selected timesteps for ddim sampler: {steps_out}')

        return steps_out
    
    def ddim_reverse(self, model, x_size, condition, only_last_sample=True, stochastic=True):
        model.eval()
        batch_size = x_size[0]
        intermediates = None
        noisy_data = 1 * stochastic * torch.randn_like(torch.zeros(x_size)).to(self.device)

        if not only_last_sample:
            intermediates = {'data_inter': [noisy_data], 'pred_data': [noisy_data]}

        time_range = np.flip(self.ddim_timesteps)
        total_steps = self.ddim_timesteps.shape[0]

        for i, step in enumerate(time_range):
            index = total_steps - i - 1
            t = torch.full((batch_size,), step, device=self.device, dtype=torch.long)

            noisy_data, reparam_data = self.ddim_step(model, noisy_data, condition, t, index)

            if not only_last_sample:
                intermediates['data_inter'].append(noisy_data)
                intermediates['pred_data'].append(reparam_data)

        if only_last_sample:
            return noisy_data
        else:
            return noisy_data, intermediates


    def ddim_step(self, model, noisy_data, condition, t, index):
        e_t = model(noisy_data, t, condition).detach()
        sqrt_one_minus_alphas = torch.sqrt(1. - self.ddim_alphas)
        # select parameters corresponding to the currently considered timestep
        a_t = torch.full(e_t.shape, self.ddim_alphas[index], device=self.device)
        a_t_m_1 = torch.full(e_t.shape, self.ddim_alphas_prev[index], device=self.device)
        sigma_t = torch.full(e_t.shape, self.ddim_sigmas[index], device=self.device)
        sqrt_one_minus_at = torch.full(e_t.shape, sqrt_one_minus_alphas[index], device=self.device)

        # direction pointing to data_t
        dir_b_t = (1. - a_t_m_1 - sigma_t ** 2).sqrt() * e_t
        noise = sigma_t * torch.randn_like(noisy_data).to(self.device)

        # reparameterize data_0
        reparam_data = (noisy_data - sqrt_one_minus_at * e_t) / a_t.sqrt()

        # compute data_t_m_1
        noisy_data_t_m_1 = a_t_m_1.sqrt() * reparam_data + 1 * dir_b_t + noise

        return noisy_data_t_m_1, reparam_data
