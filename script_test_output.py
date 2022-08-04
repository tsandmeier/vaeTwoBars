import os

import torch

from dataset.doodle_torch_dataloader import DoodleTorchDataset
from model.doodle_vae import DoodleVAE
from model.doodle_vae_trainer import DoodleVAETrainer

num_epochs = 20
batch_size = 512

dataset = DoodleTorchDataset()
vae_model = DoodleVAE(dataset)

if torch.cuda.is_available():
    vae_model.cuda()

vae_trainer = DoodleVAETrainer(dataset, vae_model)

if not os.path.exists(vae_model.filepath):
    vae_trainer.train_model(batch_size=batch_size, num_epochs=num_epochs, log=False)

vae_trainer.load_model()

vae_trainer.encode_and_decode_example()

