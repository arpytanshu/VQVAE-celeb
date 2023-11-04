
from typing import Tuple
from dataclasses import dataclass
from collections import OrderedDict

import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from torchvision.utils import make_grid


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


    

def get_ellipse(mu, std, multiplier):
    # util method to plot std_dev ellipses
    if multiplier == 1:
        color = 'r'; alpha = 0.9
    elif multiplier == 2:
        color = 'g'; alpha = 0.6
    elif multiplier == 3:
        color = 'b'; alpha = 0.3
    else:
        raise Exception('Exceeded multiplier value. Should be one of [1, 2, 3]')
    return Ellipse(xy=(mu[0], mu[1]), 
                width=multiplier*std[0], 
                height=multiplier*std[1], 
                edgecolor=color, 
                linewidth=0.5,
                alpha = alpha,
                fc='None')

def get_latent_space_plot(dataloader, model, trainer_args, save_plot=True):    
    num_batch = int(trainer_args.n_samples_visualize / dataloader.batch_size) + 1
    mu = []; log_var = []
    for _ in range(num_batch):
        images, _ = next(iter(dataloader))
        images = images.to(trainer_args.dtype).to(trainer_args.device)
        with torch.no_grad():
            m, lv = model.encode(images)
        mu.append(m); log_var.append(lv)

    mu = torch.cat(mu, dim=0).cpu()
    log_var = torch.cat(log_var, dim=0)
    std = torch.sqrt(torch.exp(log_var)).cpu()
    
    plt.figure(figsize=(15,15))
    ax = plt.gca(); 
    fig = plt.gcf()
    fig.suptitle(f"Latent Space @ Iteration{trainer_args.curr_iter}"); 
    fig.tight_layout()
    plt.xlim(left=-7, right=+7)
    plt.ylim(bottom=-7, top=+7)
    
    for ix in range(trainer_args.n_samples_visualize):
        mu_ix = mu[ix]; std_ix = std[ix]
        plt.scatter(x = mu_ix[0], y = mu_ix[1], c='k', s=1)
        ax.add_patch(get_ellipse(mu_ix, std_ix, 1))
        ax.add_patch(get_ellipse(mu_ix, std_ix, 2))
        ax.add_patch(get_ellipse(mu_ix, std_ix, 3))
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

    plt.close(fig)
    if save_plot:
        fig.savefig(trainer_args.checkpoint_path / "plots" / f"latent_space_{trainer_args.curr_iter}.png")
    else:
        return fig

def get_reconstructions_plot(dataloader, model, trainer_args, save_plot=True):
    
    norm_stats = dataloader.dataset.get_normalize_stats()
    M = torch.tensor(norm_stats['mean']).view(3,1,1)
    S = torch.tensor(norm_stats['std']).view(3,1,1)

    samples, _ = next(iter(dataloader))
    samples = samples[:64]
    
    samples = samples.to(trainer_args.dtype).to(trainer_args.device)
    recon, _, _, _ = model(samples)

    samples_grid = make_grid(samples.cpu())
    recon_grid = make_grid(recon.cpu())

    samples_grid = (samples_grid * S) + M
    recon_grid = (recon_grid * S) + M

    plt.close(fig)
    if save_plot:
        fig.savefig(trainer_args.checkpoint_path / "plots" / f"latent_space_{trainer_args.curr_iter}.png")
    else:
        return fig

    fig, axs = plt.subplots(1,2, figsize=(10, 5))
    axs[0].imshow(samples_grid.permute(1,2,0))
    axs[1].imshow(recon_grid.permute(1,2,0))
