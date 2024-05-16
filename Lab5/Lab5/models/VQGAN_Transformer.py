import torch 
import torch.nn as nn
import yaml
import os
import math
import numpy as np
import random
from .VQGAN import VQGAN
from .Transformer import BidirectionalTransformer


#TODO2 step1: design the MaskGIT model
class MaskGit(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.vqgan = self.load_vqgan(configs['VQ_Configs'])
    
        self.num_image_tokens = configs['num_image_tokens']
        self.mask_token_id = configs['num_codebook_vectors']
        self.choice_temperature = configs['choice_temperature']
        self.gamma = self.gamma_func(configs['gamma_type'])
        self.transformer = BidirectionalTransformer(configs['Transformer_param'])

    def load_transformer_checkpoint(self, load_ckpt_path):
        self.transformer.load_state_dict(torch.load(load_ckpt_path))

    @staticmethod
    def load_vqgan(configs):
        cfg = yaml.safe_load(open(configs['VQ_config_path'], 'r'))
        model = VQGAN(cfg['model_param'])
        model.load_state_dict(torch.load(configs['VQ_CKPT_path']), strict=True) 
        model = model.eval()
        return model
    
##TODO2 step1-1: input x fed to vqgan encoder to get the latent and zq
    @torch.no_grad()
    def encode_to_z(self, x):
        zq, z_ind, _ = self.vqgan.encode(x)
        return zq, z_ind
    
##TODO2 step1-2:    
    def gamma_func(self, mode="cosine"):
        """Generates a mask rate by scheduling mask functions R.

        Given a ratio in [0, 1), we generate a masking ratio from (0, 1]. 
        During training, the input ratio is uniformly sampled; 
        during inference, the input ratio is based on the step number divided by the total iteration number: t/T.
        Based on experiements, we find that masking more in training helps.
        
        ratio:   The uniformly sampled ratio [0, 1) as input.
        Returns: The mask rate (float).

        """
        if mode == "linear":
            def f(ratio):
                return 1 - ratio
            return f
        elif mode == "cosine":
            def f(ratio):
                return math.cos(math.pi * ratio / 2)
            return f
        elif mode == "square":
            def f(ratio):
                return 1 - ratio ** 2
            return f
        else:
            raise NotImplementedError

##TODO2 step1-3:            
    def forward(self, x, ratio):
        _, z_indices=self.encode_to_z(x) #ground truth
        z_indices = z_indices.view(-1, self.num_image_tokens)
        # apply mask to the ground truth
        mask = torch.bernoulli(torch.ones_like(z_indices) * ratio)
        z_indices_input = torch.where(mask == 1, torch.tensor(self.mask_token_id).to(mask.device), z_indices)
        logits = self.transformer(z_indices_input)  #transformer predict the probability of tokens
        logits = logits[..., :self.mask_token_id]
        ground_truth = torch.zeros(z_indices.shape[0], z_indices.shape[1], self.mask_token_id).to(z_indices.device).scatter_(2, z_indices.unsqueeze(-1), 1)
        return logits, ground_truth
    
##TODO3 step1-1: define one iteration decoding   
    @torch.no_grad()
    def inpainting(self, x , ratio, mask_b):
        _, z_indices=self.encode_to_z(x)
        z_indices_input = torch.where(mask_b == 1, torch.tensor(self.mask_token_id).to(mask_b.device), z_indices)
        logits = self.transformer(z_indices_input)
        #Apply softmax to convert logits into a probability distribution across the last dimension.
        logits = torch.nn.functional.softmax(logits, dim=-1)

        #FIND MAX probability for each token value
        z_indices_predict_prob, z_indices_predict = torch.max(logits, dim=-1)

        ratio=self.gamma(ratio)
        #predicted probabilities add temperature annealing gumbel noise as confidence
        g = -torch.log(-torch.log(torch.rand(1, device=z_indices_predict_prob.device))) # gumbel noise
        temperature = self.choice_temperature * (1 - ratio)
        confidence = z_indices_predict_prob + temperature * g
        
        #hint: If mask is False, the probability should be set to infinity, so that the tokens are not affected by the transformer's prediction
        #sort the confidence for the rank 
        confidence = torch.where(mask_b == 0, torch.tensor(float('inf')).to(mask_b.device), confidence)
        ratio = 0 if ratio < 1e-8 else ratio
        n = math.ceil(mask_b.sum() * ratio)
        _, idx_to_mask = torch.topk(confidence, n, largest=False)
        #define how much the iteration remain predicted tokens by mask scheduling
        #At the end of the decoding process, add back the original token values that were not masked to the predicted tokens
        mask_bc=torch.zeros_like(mask_b).scatter_(1, idx_to_mask, 1)
        torch.bitwise_and(mask_bc, mask_b, out=mask_bc)
        return z_indices_predict, mask_bc
    
__MODEL_TYPE__ = {
    "MaskGit": MaskGit
}
    


        
