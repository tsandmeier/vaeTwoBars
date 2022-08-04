import os

import torch

from dataset.doodle_torch_dataloader import DoodleTorchDataset
from model.doodle_vae import DoodleVAE
from model.doodle_vae_trainer import DoodleVAETrainer


print("Lade Dataset...")
dataset = DoodleTorchDataset()
dataset.load_dataset()

