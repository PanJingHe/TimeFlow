import math
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from einops import reduce
from tqdm.auto import tqdm
from Models.interpretable_diffusion.transformer import Transformer
from Utils.utils import visualize_components
import os



class TimeFlow(nn.Module):
    def __init__(
            self,
            seq_length,
            feature_size,
            n_layer_enc=3,
            n_layer_dec=6,
            d_model=None,
            n_heads=4,
            mlp_hidden_times=4,
            attn_pd=0.,
            resid_pd=0.,
            kernel_size=None,
            padding_size=None,
            implementation="SDE",
            **kwargs
    ):
        super(TimeFlow, self).__init__()

        self.seq_length = seq_length
        self.feature_size = feature_size

        self.model = Transformer(n_feat=self.feature_size, n_channel=seq_length, n_layer_enc=n_layer_enc, n_layer_dec=n_layer_dec,
                                 n_heads=n_heads, attn_pdrop=attn_pd, resid_pdrop=resid_pd, mlp_hidden_times=mlp_hidden_times,
                                 max_len=seq_length, n_embd=d_model, conv_params=[kernel_size, padding_size], **kwargs)
        

        self.alpha = 3  ## t shifting, change to 1 is the uniform sampling during inference
        self.time_scalar = 1000 ## scale 0-1 to 0-1000 for time embedding

        self.num_timesteps = int(os.environ.get('hucfg_num_steps', '100'))
        
        self.implementation = implementation
        
        self.sde_noise = 0.1

        

        
    def output(self, x, t, condi=None, step=0):
        

        trend,season = self.model(x, t, condi=None)
        
        output = trend + season
    
        return output



    @torch.no_grad()
    def sample(self, shape):
        
        
        self.eval()

        zt = torch.randn(shape).cuda()  ## init the noise 

        ## t shifting from stable diffusion 3
        timesteps = torch.linspace(0, 1, self.num_timesteps+1)
        t_shifted = 1-(self.alpha * timesteps) / (1 + (self.alpha - 1) * timesteps)
        t_shifted = t_shifted.flip(0)

        for t_curr, t_prev in zip(t_shifted[:-1], t_shifted[1:]):
            step = t_prev - t_curr
            v = self.output(zt.clone(), torch.tensor([t_curr*self.time_scalar]).unsqueeze(0).repeat(zt.shape[0], 1).cuda().squeeze())                  
            zt = zt.clone() + step * v 

        return zt
    
    @torch.no_grad()
    def sde_sample(self, shape):
        
        sde = SDE(self.model, noise=self.sde_noise)
        
        self.eval()

        zt = torch.randn(shape).cuda()  ## init the noise 

        timesteps = torch.linspace(0, 1, self.num_timesteps+1)
        t_shifted = 1-(self.alpha * timesteps) / (1 + (self.alpha - 1) * timesteps)
        t_shifted = t_shifted.flip(0)

        for t_curr, t_prev in zip(t_shifted[:-1], t_shifted[1:]):
            step = t_prev - t_curr
            v = self.output(zt.clone(), torch.tensor([t_curr*self.time_scalar]).unsqueeze(0).repeat(zt.shape[0], 1).cuda().squeeze()) 
            diffusion=torch.ones_like(t_prev) * torch.ones_like(t_prev) * self.sde_noise
            noise = torch.randn_like(zt) * torch.sqrt(step)                 
            zt = zt.clone() + step * v + diffusion * noise

        return zt

    
    
    def generate_mts(self, batch_size):
        """
        生成 shape = [batch_size, seq_len, feature_dim] 的时间序列
        """

        feature_dim = self.feature_size
        seq_len = self.seq_length
        
        if self.implementation == "ODE":
            return self.sample((batch_size, seq_len, feature_dim))
        elif self.implementation == "SDE":
            return self.sde_sample((batch_size, seq_len, feature_dim))
        
    

    
    def compute_loss(self, x_start, step=0):
        # z0: noise / prior
        z0 = torch.randn_like(x_start)  
        z1 = x_start  # target

        t = torch.rand(z0.shape[0], 1, 1).to(z0.device)

        z_t = t * z1 + (1. - t) * z0  # [B, ..., D]
        target = z1 - z0              # [B, ..., D]
        

        model_out = self.output(z_t, t.squeeze() * self.time_scalar, condi=None, step=step)  # [B, ..., D + 1]


        if self.implementation == "SDE":
            variance = t * (1 - t) * (self.sde_noise ** 2)
            noise = torch.randn_like(model_out) * torch.sqrt(variance)
            model_out = model_out + noise
        
        mse_loss = F.mse_loss(model_out, target, reduction='none')  # [B, ..., D]
        mse_loss = reduce(mse_loss, 'b t d -> b t', 'mean')
        # reduce feature dim
        total_loss = mse_loss.mean()

        return total_loss


    def forward(self, x, step, logger=None):
        b, c, n, device, feature_size, = *x.shape, x.device, self.feature_size
        assert n == feature_size, f'number of variable must be {feature_size}'
        return self.compute_loss(x,step=step)

    def fast_sample_infill(self, shape, target, partial_mask=None):

        z0 = torch.randn(shape).cuda()
        z1 = zt = z0
        for t in range(self.num_timesteps):
            t = t/self.num_timesteps  ## scale to 0-1
            t = t**(float(os.environ.get('hucfg_Kscale', 0.03)))  ## perform t-power sampling

            
            z0 = torch.randn(shape).cuda()  ## re init the z0

            target_t = target*t + z0*(1-t)  ## get the noisy target
            zt = z1*t + z0*(1-t)  ##
            # import ipdb; ipdb.set_trace()
            zt[partial_mask] = target_t[partial_mask]  ## replace with the noisy version of given ground truth information
            #diffusion=torch.ones_like(torch.tensor([t]).cuda())*torch.ones_like(torch.tensor([t]).cuda()) * self.sde_noise
            # noise = torch.randn_like(zt) * torch.sqrt(torch.tensor([1-t]).cuda()) 
            v = self.output(zt, torch.tensor([t*self.time_scalar]).cuda(), None) 
            z1 = zt.clone() + (1 - t) * v  ## one step euler
            z1 = torch.clamp(z1, min=-1, max=1) ## make sure the upper and lower bound dont exceed


        return z1





class SDE(torch.nn.Module):

    # noise is sigma in this notebook for the equation sigma * (t * (1 - t))
    def __init__(self, ode_drift, noise=1.0, reverse=False):
        super().__init__()
        self.reverse = reverse
        self.noise = noise

    # Drift
    def f(self, y):
        return self.drift(y)

    # Diffusion
    def g(self, t, y):
        return torch.ones_like(t) * torch.ones_like(y) * self.noise




