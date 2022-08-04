from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from dataset.doodle_dataset import DoodleDataset


class DoodleTorchDataset:
    def __init__(self):
        self.kwargs = {'pin_memory': True} if torch.cuda.is_available() else {}
        self.dataset = None
        self.note2index_dict = None
        self.index2note_dict = None
        self.latent_dicts = None
        self.tick_durations = None
        self.beat_subdivisions = None

    def load_dataset(self):
        dataset = DoodleDataset()
        dataset.make_or_load_dataset()
        score = dataset.score_array
        score = np.expand_dims(score, axis=1)
        latent_values = dataset.latent_array
        a = np.c_[
            score.reshape(len(score), -1),
            latent_values.reshape(len(latent_values), -1)
        ]
        score2 = a[:, :score.size // len(score)].reshape(score.shape)
        latent_values2 = a[:, score.size // len(score):].reshape(latent_values.shape)
        np.random.shuffle(a)

        for index, eintrag in enumerate(score2):
            score2[index] = eintrag.tolist()

        self.dataset = TensorDataset(
            torch.from_numpy(score2),
            torch.from_numpy(latent_values2)
        )
        self.note2index_dict = dataset.note2index_dict
        self.index2note_dict = dataset.index2note_dict
        self.latent_dicts = dataset.latent_dicts
        self.beat_subdivisions = dataset.beat_subdivisions
        self.tick_durations = dataset.tick_durations

    def data_loaders(
            self, batch_size: int, split: tuple = (0.70, 0.20)  # 0.7, 0.2
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Returns three data loaders obtained by splitting the data
        Args:
            batch_size: int, number of data points in each batch
            split: tuple, specify the ratio in which the dataset is to be divided
        Returns:
            tuple of 3 DataLoader objects corresponding to the train, validation and test sets
        """
        assert sum(split) < 1

        if self.dataset is None:
            self.load_dataset()

        num_examples = len(self.dataset)
        a, b = split
        train_dataset = TensorDataset(
            *self.dataset[: int(a * num_examples)]
        )
        val_dataset = TensorDataset(
            *self.dataset[int(a * num_examples):int((a + b) * num_examples)]
        )
        eval_dataset = TensorDataset(
            *self.dataset[int((a + b) * num_examples):]
        )

        train_dl = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            **self.kwargs
        )

        val_dl = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=True,
        )

        eval_dl = DataLoader(
            eval_dataset,
            batch_size=batch_size,
            shuffle=False,
        )
        return train_dl, val_dl, eval_dl

    def get_score(self, index):

        if self.dataset is None:
            self.load_dataset()

        return self.dataset[index][0], self.dataset[index][1]

