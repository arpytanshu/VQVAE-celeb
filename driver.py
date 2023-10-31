#%%

from data.celeb_utils import get_dataset
from pathlib import Path
import matplotlib.pyplot as plt
import torch
from torchvision.transforms import Resize
import numpy as np
from model.VAE import VAEModel, VAEConfig
from trainer import TrainingArgs, Trainer






args = TrainingArgs()
dataset_path = Path('/shared/datasets/Celeb-A')
args.dataset_path = dataset_path

trainer = Trainer(args=args)
trainer.train()




# %%
