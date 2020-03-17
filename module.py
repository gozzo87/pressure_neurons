import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from torch.optim import *

###############################################################################
#
# Split Block Abstract Class
#
###############################################################################


class SpModule(nn.Module):

  def __init__(self, can_split=True, actv_fn='relu', has_bn=False):
    super(SpModule, self).__init__()
    self.can_split = can_split
    self.actv_fn = actv_fn
    self.has_bn = has_bn
    self.bn = None
    self.w = None
    self.v = None
    self.Ys = []
    self.module = None
    self.leaky_alpha = 0.2

  def _reset_Ys(self):
    for _ in self.Ys:
      del _
    self.Ys = []

  def _d2_actv(self, x, beta=3.):
    if self.actv_fn == 'relu':
      # use 2nd order derivative of softplus for approximation
      s = torch.sigmoid(x * beta)
      return beta * s * (1. - s)
    elif self.actv_fn == 'softplus':
      s = torch.sigmoid(x * beta)
      return s * (1. - s)
    elif self.actv_fn == 'rbf':
      return (x.pow(2) - 1) * (-x.pow(2) / 2).exp()
    elif self.actv_fn == 'leaky_relu':
      s = torch.sigmoid(x * beta)
      return beta * s * (1. - s) * (1. - self.leaky_alpha)
    elif self.actv_fn == 'swish':
      s = torch.sigmoid(x)
      return s * (1. - s) + s + x * s * (1. - s) - (
          s.pow(2) + 2. * x * s.pow(2) * (1. - s))
    elif self.actv_fn == 'sigmoid':
      s = torch.sigmoid(x)
      return (s - s.pow(2)) * (1. - s).pow(2)
    elif self.actv_fn == 'tanh':
      h = torch.tanh(x)
      return -2. * h * (1 - h.pow(2))
    elif self.actv_fn == 'none':
      return torch.ones_like(x)
    else:
      raise Exception('[ERROR] unknown activation')

  def _activate(self, x):
    if self.actv_fn == 'relu':
      return F.relu(x)
    elif self.actv_fn == 'leaky_relu':
      return F.leaky_relu(x, self.leaky_alpha)
    elif self.actv_fn == 'rbf':
      return (-x.pow(2) / 2).exp()
    elif self.actv_fn == 'swish':
      return x * torch.sigmoid(x)
    elif self.actv_fn == 'sigmoid':
      return torch.sigmoid(x)
    elif self.actv_fn == 'tanh':
      return torch.tanh(x)
    if self.actv_fn == 'softplus':
      return F.softplus(x)
    elif self.actv_fn == 'none':
      return x
    else:
      raise Exception('[ERROR] unknown activation')

  def sp_noise(self):
    pass

  def sp_delete_noise(self):
    del self.noise

  def sp_eigen(self, avg_over=1.):
    pass

  def forward(self, x, pre_split=False):
    pass

  def sp_forward(self, x):
    """
        the forward pass that determines the splitting matrix.
        """
    if self.has_bn:
      self.bn.eval()
    pass

  def active_split(self, threshold):
    """
        actively split the current layer.
        @param threshold: neurons w/ eigenvalues above threshold should split.
        """
    pass

  def passive_split(self, idx):
    """
        passively split due to the splitting of a previous layer.
        @param idx: which dimensions should be duplicated.
        """
    pass
