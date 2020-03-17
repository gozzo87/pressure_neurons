import numpy as np
import torch
import torch.nn as nn
from torch.optim import *


class SpNet(nn.Module):

  def __init__(self):
    super(SpNet, self).__init__()
    self.net = None
    self.n_elites = 1
    self.next_layers = {}
    self.layers_to_split = []

  def create_optimizer(self):
    pass

  def forward(self, x, split_layer=-1):
    pass

  def loss_fn(self, split_layer=-1):
    pass

  def update(self, x, y):
    pass

  def split(self):
    for layer in self.net:
      layer._reset_Ys()

  def get_pressure(self):
    w_list = [self.net[layer].w for layer in self.layers_to_split]
    return np.concatenate(w_list).reshape(-1)

  def sp_where(self):
    w_list = self.get_pressure()
    idx = min(len(w_list[w_list < 0]) - 1, self.n_elites - 1)
    if idx < 0:
      return -1000
    else:
      return w_list[idx]

  def get_num_params(self):
    model_n_params = sum(
        p.numel() for p in self.parameters() if p.requires_grad)
    return model_n_params
