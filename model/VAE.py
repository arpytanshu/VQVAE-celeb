#%%
import torch
from torch import nn
import torch.nn.functional as F

from typing import Tuple
from dataclasses import dataclass
from collections import OrderedDict

@dataclass
class VAEConfig():
    latent_dim: int = 2
    in_channel: int = 3
    hidden_dims: Tuple = (32, 64, 128, 256, 512)
    kernel_sz: Tuple = 3
    stride: int = 2
    padding: int = 1
    enc_out_shape: Tuple = (7, 6)
    linear_mul: int = enc_out_shape[0] * enc_out_shape[1]
    

class VAEModel(nn.Module):
    def __init__(self, cfg: VAEConfig):
        super().__init__()
        self.cfg = cfg
        
        self.encoder = self.build_encoder()

        # linear layers to project to latent dimension
        self.linear_mu = nn.Linear(self.cfg.linear_mul*self.cfg.hidden_dims[-1],
                                   self.cfg.latent_dim)
        self.linear_var = nn.Linear(self.cfg.linear_mul*self.cfg.hidden_dims[-1],
                                   self.cfg.latent_dim)

        self.decoder_input = nn.Linear(self.cfg.latent_dim,
                                       self.cfg.linear_mul*self.cfg.hidden_dims[-1])
        
        self.decoder = self.build_decoder()

    def forward(self, input):
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), z, mu, log_var
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu

    def get_num_params(self):
        n_params = sum(p.numel() for p in self.parameters())
        return f"{n_params // 1024 // 1024}M"

    def encode(self, input):
        x = self.encoder(input)
        x = x.view(x.size(0), -1)
        mu = self.linear_mu(x)
        log_var = self.linear_var(x)
        return mu, log_var

    def decode(self, z):
        z = self.decoder_input(z)
        z = z.view(z.size(0), -1, self.cfg.enc_out_shape[0], self.cfg.enc_out_shape[1])
        return self.decoder(z)

    def build_encoder(self):
        in_channel = self.cfg.in_channel
        layers = []
        for out_channel in self.cfg.hidden_dims:
            module = nn.Sequential(OrderedDict([
                ('conv2d', nn.Conv2d(in_channels=in_channel,
                                     out_channels=out_channel,
                                     kernel_size=self.cfg.kernel_sz,
                                     stride=self.cfg.stride,
                                     padding=self.cfg.padding)),
                ('bn', nn.BatchNorm2d(out_channel)),
                ('leakyrelu', nn.LeakyReLU())
                ]))
            layers.append(module)
            in_channel = out_channel
        return nn.Sequential(*layers)
    
    def build_decoder(self):
        layers = []

        in_channel = self.cfg.hidden_dims[-1]
        for out_channel in self.cfg.hidden_dims[:-1][::-1]:
            module = nn.Sequential(OrderedDict([
                        ('convT2d', nn.ConvTranspose2d(in_channels=in_channel,
                                                       out_channels=out_channel,
                                                       kernel_size=self.cfg.kernel_sz,
                                                       stride=self.cfg.stride,
                                                       padding=self.cfg.padding,
                                                       output_padding=(1,1)
                                                       )),
                        ('bn', nn.BatchNorm2d(out_channel)),
                        ('leakyrelu', nn.LeakyReLU())
                    ]))
            layers.append(module)
            in_channel = out_channel
        
        final_layer = nn.Sequential(OrderedDict([
            ('convT2d', nn.ConvTranspose2d(self.cfg.hidden_dims[0],
                                           self.cfg.hidden_dims[0],
                                           kernel_size=3,
                                           stride=2,
                                           padding=1,
                                           output_padding=1
                                           )),
            ('bn', nn.BatchNorm2d(self.cfg.hidden_dims[0])),
            ('leakyrelu', nn.LeakyReLU()),
            ('final_conv', nn.Conv2d(self.cfg.hidden_dims[0], 
                                     out_channels= 3, 
                                     kernel_size= 3, 
                                     padding=1)),
            ('tanh', nn.Tanh())
            ]))
        layers.append(final_layer)
        return nn.Sequential(*layers)

    
def VAELoss(input, reconst, mu, log_var, kld_weight):
    
    recons_loss = F.mse_loss(reconst, input)

    kld_loss = torch.mean(-0.5 * \
                            torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1),
                            dim = 0)

    loss = recons_loss + (kld_weight * kld_loss)
    
    return {'loss': loss, 'recon_loss':recons_loss.detach(), 'kld':-kld_loss.detach()}