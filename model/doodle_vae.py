import os

import torch
from torch.autograd import Variable

from model.decoder import HierarchicalDecoder
from model.encoder import Encoder
from model.model import Model
from torch import nn


class DoodleVAE(Model):
    def __init__(self,
                 dataset,
                 note_embedding_dim=10,
                 metadata_embedding_dim=2,
                 encoder_hidden_size=64,
                 encoder_dropout_prob=0.2,
                 num_encoder_layers=2,
                 latent_space_dim=64,  # vorsicht! Geändert
                 num_decoder_layers=2,
                 decoder_hidden_size=64,
                 decoder_dropout_prob=0.2
                 ):
        super(DoodleVAE, self).__init__()
        
        self.num_beats_per_measure = 4
        self.num_ticks_per_measure = 16
        self.num_ticks_per_beat = int(self.num_ticks_per_measure / self.num_beats_per_measure)

        self.num_notes = 64

        self.note_embedding_dim = note_embedding_dim
        self.metadata_embedding_dim = metadata_embedding_dim
        self.num_encoder_layers = num_encoder_layers
        self.encoder_hidden_size = encoder_hidden_size
        self.encoder_dropout_prob = encoder_dropout_prob
        self.latent_space_dim = latent_space_dim
        self.num_decoder_layers = num_decoder_layers
        self.decoder_hidden_size = decoder_hidden_size
        self.decoder_dropout_prob = decoder_dropout_prob
        # self.has_metadata = has_metadata

        cur_dir = os.path.dirname(os.path.realpath(__file__))
        self.filepath = os.path.join(
            os.path.dirname(cur_dir),
            'saved_models',
            'modelbeta.pt'
        )

        # Encoder
        self.encoder = Encoder(
            note_embedding_dim=self.note_embedding_dim,
            rnn_hidden_size=self.encoder_hidden_size,
            num_layers=self.num_encoder_layers,
            num_notes=self.num_notes,
            dropout=self.encoder_dropout_prob,
            bidirectional=True,
            z_dim=self.latent_space_dim,
            rnn_class=nn.GRU
        )

        self.decoder = HierarchicalDecoder(
            note_embedding_dim=self.note_embedding_dim,
            num_notes=self.num_notes,
            z_dim=self.latent_space_dim,
            num_layers=self.num_decoder_layers,
            rnn_hidden_size=self.decoder_hidden_size,
            dropout=self.decoder_dropout_prob,
            rnn_class=nn.GRU
        )

    def __repr__(self):
        return 'Doodle_VAE'

    def forward(self, measure_score_tensor: Variable,
                measure_metadata_tensor: Variable, train=True):
        """
        Implements the forward pass of the VAE
        :param measure_score_tensor: torch Variable,
                (batch_size, measure_seq_length)
        :param measure_metadata_tensor: torch Variable,
                (batch_size, measure_seq_length, num_metadata)
        :param train: bool,
        :return: torch Variable,
                (batch_size, measure_seq_length, self.num_notes)
        """
        # check input
        seq_len = measure_score_tensor.size(1)
        assert (seq_len == 32)
        # compute output of encoding layer
        z_dist = self.encoder(measure_score_tensor)

        # sample from distribution
        z_tilde = z_dist.rsample()

        # compute prior
        prior_dist = torch.distributions.Normal(
            loc=torch.zeros_like(z_dist.loc),
            scale=torch.ones_like(z_dist.scale)
        )
        z_prior = prior_dist.sample()

        # compute output of decoding layer
        weights, samples = self.decoder(
            z=z_tilde,
            score_tensor=measure_score_tensor,
            train=train
        )
        return weights, samples, z_dist, prior_dist, z_tilde, z_prior

    # def forward(selfself):
    #     pass
