import os

import torch
from torch import nn


class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()

    def forward(selfself):
        raise NotImplementedError

    def xavier_initialization(self):
        """
        Initializes the network params
        :return:
        """
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)

    def save(self):
        """
        Saves the src
        :return: None
        """
        save_dir = os.path.dirname(self.filepath)
        # create save directory if needed
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        torch.save(self.state_dict(), self.filepath)
        print(f'Model {self.__repr__()} saved')
        # print("SaveDir: ", save_dir)

    def load(self, cpu=False):
        """
        Loads the src
        :param cpu: bool, specifies if the src should be loaded on the CPU
        :return: None
        """
        if cpu:
            self.load_state_dict(
                torch.load(
                    self.filepath,
                    map_location=lambda storage,
                    loc: storage
                )
            )
        else:
            self.load_state_dict(torch.load(self.filepath))
        # print(f'Model {self.__repr__()} loaded')
