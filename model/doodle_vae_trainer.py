import numpy as np
import torch

from dataset.helpers import convert_tensor_to_music_21
from model.doodle_vae import DoodleVAE
from model.trainer import Trainer
from utils.helpers import to_cuda_variable_long, to_cuda_variable


class DoodleVAETrainer(Trainer):
    def __init__(self,
                 dataset,
                 model: DoodleVAE,
                 model_type='beta-VAE',
                 lr=1e-4,
                 beta=0.001,
                 gamme=1.0,
                 delta=10.0,
                 capacity=0.0,
                 device=0,
                 rand=0
                 ):
        super(DoodleVAETrainer, self).__init__(dataset, model, lr)
        self.model_type = model_type
        self.metrics = {}
        self.beta = beta
        self.capacity = capacity
        self.cur_epoch_num = 0
        self.warm_up_epochs = 10
        self.num_iterations = 100000

        if self.model_type == 'beta-VAE':
            self.exp_rate = np.log(1 + self.beta) / self.num_iterations
            self.start_beta = 0.0
            self.cur_beta = self.start_beta
            self.start_capacity = self.capacity
            self.cur_capacity = self.capacity

        self.anneal_iterations = 0
        self.device = device
        self.rand_seed = rand
        torch.manual_seed(self.rand_seed)
        np.random.seed(self.rand_seed)

    def loss_and_acc_for_batch(self, batch, epoch_num=None, batch_num=None, train=True):
        if self.cur_epoch_num != epoch_num:
            flag = True
            self.cur_epoch_num = epoch_num
        else:
            flag = False

        score, latent_attributes = batch

        weights, samples, z_dist, prior_dist, z_tilde, z_prior = self.model(
            measure_score_tensor=score,
            measure_metadata_tensor=None,
            train=train
        )

        #compute reconstruction loss
        recons_loss = self.reconstruction_loss(x=score, x_recons=weights)

        # compute KLD loss
        dist_loss = self.compute_kld_loss(z_dist, prior_dist, beta=self.cur_beta, c=0.0)
        dist_loss = torch.nn.functional.relu(dist_loss - self.cur_capacity)

        loss = recons_loss + dist_loss

        accuracy = self.mean_accuracy(
            weights=weights, targets=score
        )

        return loss, accuracy

    def process_batch_data(self, batch):
        """
        Processes the batch returned by the dataloader iterator
        :param batch: object returned by the dataloader iterator
        :return: tuple of Torch Variable objects
        """
        score_tensor, latent_tensor = batch
        # convert input to torch Variables
        batch_data = (
            to_cuda_variable_long(score_tensor.squeeze(1), self.device),
            to_cuda_variable_long(latent_tensor.squeeze(1), self.device)
        )
        return batch_data

    @staticmethod
    def reconstruction_loss(x, x_recons):
        return Trainer.mean_crossentropy_loss(weights=x_recons, targets=x)

    @staticmethod
    def compute_reg_loss(z, labels, reg_dim, gamma, factor=1.0):
        """
        Computes the regularization loss
        """
        x = z[:, reg_dim]
        reg_loss = Trainer.reg_loss_sign(x, labels, factor=factor)
        return gamma * reg_loss

    def encode_score(self, score):
        z_dist = self.model.encoder(to_cuda_variable(score))
        return z_dist

    def decode_latent_code(self, latent_code):
        dummy_score_tensor = to_cuda_variable(
            torch.zeros(1, 32)
        )
        _, tensor_score = self.model.decoder(to_cuda_variable(latent_code), dummy_score_tensor, False)

        return tensor_score

    def encode_and_decode_example(self):
        score, lat_attr = self.dataset.get_score(200)

        z_dist = self.encode_score(score)

        z = z_dist.rsample()

        score_decoded = self.decode_latent_code(z)

        print(score_decoded)

        bla = convert_tensor_to_music_21(score_decoded)

        # print("B ", b)
