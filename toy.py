
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import datasets, transforms

from . import linear
from . import module
from . import net

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class net(net.SpNet):

  def __init__(self, n_neurons=3):
    super(net, self).__init__()
    self.n_neurons = n_neurons
    self.verbose = False
    self.net = nn.ModuleList([
        linear.Linear(2, self.n_neurons, bias=False, actv_fn='rbf'),
        linear.Linear(
            self.n_neurons, 1, can_split=False, bias=False, actv_fn='none')
    ])
    # self.net[0].module.weight.data = torch.randn_like(self.net[0].module.weight)
    # self.net[1].module.weight.data = torch.randn_like(self.net[1].module.weight)
    self.next_layers = {
        0: [1],
    }
    self.layers_to_split = list(self.next_layers.keys())
    self.n_elites = 1
    self.create_optimizer()

  def create_optimizer(self):
    self.opt = Adam(self.net.parameters(), lr=0.01)

  def forward(self, x, fast=False, split=False):
    x = torch.cat((x, torch.ones_like(x)), -1)  # [B, 2]
    if split:
      x = self.net[0](x, True)
    elif fast:
      x = self.net[0](x, True, True)
    else:
      x = self.net[0](x)

    x = self.net[1](x)
    return x

  def compute_pressure(self, x, y):
    self.opt.zero_grad()
    y_hat = self.forward(x, split=True)
    loss = F.mse_loss(y, y_hat)
    loss.backward()
    for i in range(1):
      self.net[i].sp_eigen()
    for layer in self.net:
      layer._reset_Ys()
    return self.get_pressure()

  def update(self, x, y):
    self.opt.zero_grad()
    y_hat = self.forward(x)
    loss = F.mse_loss(y, y_hat)
    loss.backward()
    self.opt.step()
    return loss.detach().cpu().item()

  def split(self, x, y, epsilon):
    start_time = time.time()
    self.opt.zero_grad()
    y_hat = self.forward(x, split=True)
    loss = F.mse_loss(y, y_hat)
    loss.backward()
    for i in range(1):
      self.net[i].sp_eigen()
    threshold = self.sp_where()

    n_neurons = []
    n_neurons_added = {}
    # -- actual splitting -- #
    for i in reversed(self.layers_to_split):  # i is the i-th module in self.net
      n_new, idx = self.net[i].active_split(threshold, epsilon)
      n_neurons_added['classifier layer %d' % i] = n_new
      n_neurons.append(n_new)
      if n_new > 0:  # we have indeed splitted this layer
        for j in self.next_layers[i]:
          self.net[j].passive_split(idx)

    # don't forget to clean up auxiliaries
    for layer in self.net:
      layer._reset_Ys()

    # re-new optimizer
    self.create_optimizer()
    end_time = time.time()
    if self.verbose:
      print(
          '[INFO] splitting takes %10.4f sec. Threshold eigenvalue is %10.4f' %
          (end_time - start_time, threshold))
      print('[INFO] number of added neurons: \n%s\n' % '\n'.join([
          '-- %s grows %d neurons' % (x, y) for x, y in n_neurons_added.items()
      ]))
    return n_neurons

  def fast_split(self, x, y, epsilon):
    start_time = time.time()

    for i in self.layers_to_split:
      self.net[i].sp_noise()

    granularity = 5
    alphas = np.linspace(0, 1, granularity * 2)
    for alpha in alphas[1::2]:
      y_hat = self.forward(x, fast=True)
      loss = F.mse_loss(y, y_hat)
      loss.backward()
      for i in self.layers_to_split:
        self.net[i].sp_cumulate_reimann_sum(granularity)

    # -- calculate the cutoff threshold for determining whom to split -- #
    threshold = self.sp_where()

    n_neurons_added = {}
    n_neurons = []
    # -- actual splitting -- #
    for i in reversed(self.layers_to_split):  # i is the i-th module in self.net
      n_new, idx = self.net[i].active_split(threshold, epsilon)
      n_neurons_added['classifer layer %d' % i] = n_new
      n_neurons.append(n_new)
      if n_new > 0:  # we have indeed splitted this layer
        for j in self.next_layers[i]:
          self.net[j].passive_split(idx)

    # don't forget to clean up auxiliaries
    for layer in self.net:
      layer._reset_Ys()

    # re-new optimizer
    self.create_optimizer()
    end_time = time.time()
    if self.verbose:
      print('[INFO] splitting takes %10.4f sec. Threshold value is %10.9f' %
            (end_time - start_time, threshold))
      print('[INFO] number of added neurons: \n%s\n' % '\n'.join([
          '-- %s grows %d neurons' % (x, y) for x, y in n_neurons_added.items()
      ]))
    return n_neurons

  def random_split(self, epsilon):
    start_time = time.time()
    n_neurons_added = {}
    n_layers_to_split = len(self.layers_to_split)
    n_news = [self.n_elites]

    n_neurons_added = {}
    n_neurons = []
    # -- actual splitting -- #
    for i in reversed(self.layers_to_split):  # i is the i-th module in self.net
      n_new, idx = self.net[i].random_split(n_news[i], epsilon)
      n_neurons_added['classifer layer %d' % i] = n_new
      n_neurons.append(n_new)
      if n_new > 0:  # we have indeed splitted this layer
        for j in self.next_layers[i]:
          self.net[j].passive_split(idx)

    # don't forget to clean up auxiliaries
    for layer in self.net:
      layer._reset_Ys()

    # re-new optimizer
    self.create_optimizer()
    end_time = time.time()
    if self.verbose:
      print('[INFO] random splitting takes %10.4f sec.' %
            (end_time - start_time))
      print('[INFO] number of added neurons: \n%s\n' % '\n'.join([
          '-- %s grows %d neurons' % (x, y) for x, y in n_neurons_added.items()
      ]))
    return n_neurons
