#%%

from data.celeb_utils import get_dataset
from pathlib import Path
import matplotlib.pyplot as plt
from torchvision.transforms import Resize
import numpy as np
from model.VAE import VAEModel, VAEConfig
from trainer import TrainingArgs, Trainer




args = TrainingArgs()
dataset_path = Path('/shared/datasets/Celeb-A')
args.dataset_path = dataset_path

# trainer = Trainer(args=args)
trainer = Trainer(checkpoint_path='runs/231031-1757', train_batch_sz=192)

trainer.train()


# %%
