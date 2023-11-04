#%%

from data.celeb_utils import get_dataset
from pathlib import Path
import matplotlib.pyplot as plt
import torch
from torchvision.transforms import Resize
import numpy as np
from model.VAE import VAEModel, VAEConfig
from trainer import TrainingArgs, Trainer
from torchvision.utils import make_grid





args = TrainingArgs()
dataset_path = Path('/shared/datasets/Celeb-A')
args.dataset_path = dataset_path

trainer = Trainer(args=args)
# trainer = Trainer(checkpoint_path='runs/231031-2251')








#%%

anchor_point = [-3, -3]

samples = []
for x in np.linspace(anchor_point[0]-0.5, anchor_point[0]+0.5, 8):
    for y in np.linspace(anchor_point[1]-0.5, anchor_point[1]+0.5, 8):
        samples.append([x, y])

samples = torch.tensor(samples).to(trainer.args.dtype).to(trainer.args.device)

output = trainer.model.decode(samples)
output = output.detach().cpu()
image = make_grid(output)

image = (image * S) + M

plt.imshow(image.permute(1,2,0))

# %% plot reconstructions

M = torch.tensor([0.5061, 0.4254, 0.3828]).view(3,1,1)
S = torch.tensor([0.3105, 0.2903, 0.2896]).view(3,1,1)


samples, _ = next(iter(trainer.te_dataloader))
samples = samples[:64]
samples = samples.to(trainer.args.dtype).to(trainer.args.device)

recon, _, _, _ = trainer.model(samples)

print(recon.shape)

samples_grid = make_grid(samples.cpu())
recon_grid = make_grid(recon.cpu())

samples_grid = (samples_grid * S) + M
recon_grid = (recon_grid * S) + M

fig, axs = plt.subplots(1,2, figsize=(10, 5))
axs[0].imshow(samples_grid.permute(1,2,0))
axs[1].imshow(recon_grid.permute(1,2,0))


# %%


# %%
