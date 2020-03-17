from .module import SpModule
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.optim import *


class Linear(SpModule):

  def __init__(self,
               in_features,
               out_features,
               bias=True,
               can_split=True,
               actv_fn='relu',
               has_bn=False):

    super(Linear, self).__init__(
        can_split=can_split, actv_fn=actv_fn, has_bn=has_bn)

    self.has_bias = bias
    if has_bn:
      self.bn = nn.BatchNorm1d(out_features)
      self.has_bias = False

    self.module = nn.Linear(in_features, out_features, self.has_bias)
    self.w = np.zeros(out_features)

  def forward(self, x, pre_split=False, fast_forward=False, alpha=0):
    if pre_split and self.can_split:
      if fast_forward:
        return self.sp_fast_forward(x, alpha)
      else:
        return self.sp_forward(x)
    x = self.module(x)
    if self.has_bn:
      x = self.bn(x)
    x = self._activate(x)
    return x

  def sp_noise(self):
    dim_out, dim_in = self.module.weight.data.shape
    self.w = np.zeros(dim_out)
    self.noise = nn.Linear(dim_in, dim_out, bias=False).cuda()
    # self.noise.weight.data = F.normalize(self.noise.weight.data, p=2, dim=-1)
    self.v = self.noise.weight.data.cpu().numpy()

  def sp_fast_forward(self, x, alpha=0, epsilon=1e-2):
    if self.has_bn:
      self.bn.eval()
    coef = alpha * epsilon
    noise_out = self.noise(x) * coef
    out = self.module(x)
    out_plus = out + noise_out
    out_minus = out - noise_out
    if self.has_bn:
      out_plus = self.bn(out_plus)
      out_minus = self.bn(out_minus)
    out_plus = self._activate(out_plus)
    out_minus = self._activate(out_minus)
    return (out_plus + out_minus) / 2

  def sp_cumulate_reimann_sum(self, d):
    grad = self.noise.weight.grad / d
    attr = (grad * self.noise.weight.data).sum(-1).detach().cpu().numpy()
    self.w += attr

  def sp_forward(self, x):
    """
        Notice that for a single neuron:
            tr(Y nabla2_(sigma(x))) = tr(Y nabla2(sigma) xx^T) = nabla2(sigma)
            x^TYx
            nabla2 denote the 2nd order derivative
        x: [B, H_in]
    """
    out = self.module(x)  # [B, H_out]
    if self.has_bn:
      self.bn.eval()  # fix running mean/variance
      out = self.bn(out)
      # calculate bn_coff
      bn_coff = 1. / torch.sqrt(self.bn.running_var + 1e-5) * self.bn.weight
      bn_coff = bn_coff.view(1, -1)  # [1, n_out]

    first_run = (len(self.Ys) == 0)

    # calculate 2nd order derivative of the activation
    nabla2_out = self._d2_actv(out)  # [B, H_out]
    B, H_in = x.shape
    H_out = out.shape[1]
    device = 'cuda' if self.module.weight.is_cuda else 'cpu'
    auxs = []  # separate calculations for each neuron for space efficiency
    for neuron_idx in range(H_out):
      c = bn_coff[:, neuron_idx:neuron_idx + 1] if self.has_bn else 1.
      if first_run:
        Y = Variable(
            torch.zeros(H_in, H_in).to(device),
            requires_grad=True)  # [H_in, H_in]
        self.Ys.append(Y)
      else:
        Y = self.Ys[neuron_idx]
      aux = c * x.mm(Y).unsqueeze(1).bmm(c * x.unsqueeze(-1)).squeeze(
          -1)  # (Bx)Y(Bx^T), [B, 1]
      auxs.append(aux)

    auxs = torch.cat(auxs, -1)  # [B, H_out]
    auxs = auxs * nabla2_out  # [B, H_out]
    out = self._activate(out) + auxs
    return out

  def sp_eigen(self, avg_over=1.):
    A = np.array([item.grad.data.cpu().numpy() for item in self.Ys
                 ])  # [n_neurons, H_in, H_in]
    A /= avg_over
    A = (A + np.transpose(A, [0, 2, 1])) / 2
    w, v = np.linalg.eig(A)  # [H_out, K], [H_out, H_in, K]
    w = np.real(w)
    v = np.real(v)
    min_idx = np.argmin(w, axis=1)
    w_min = np.min(w, axis=1)  # [H_out,]
    v_min = v[np.arange(w_min.shape[0]), :, min_idx]  # [H_out, H_in]
    self.w = w_min
    self.v = v_min

  def random_split(self, n, epsilon=1e-2):
    H_new = n
    if H_new == 0:
      return 0, None

    H_out, H_in = self.module.weight.shape
    idx = np.random.choice(H_out, H_new)
    device = 'cuda' if self.module.weight.is_cuda else 'cpu'

    delta1 = torch.randn(H_new, H_in).to(device)
    delta2 = torch.randn(H_new, H_in).to(device)
    idx = torch.LongTensor(idx).to(device)

    new_layer = nn.Linear(H_in, H_out + H_new, bias=self.has_bias).to(device)

    # for current layer
    new_layer.weight.data[:H_out, :] = self.module.weight.data.clone()
    new_layer.weight.data[H_out:, :] = self.module.weight.data[idx, :]
    new_layer.weight.data[idx, :] += epsilon * delta1
    new_layer.weight.data[H_out:, :] -= epsilon * delta2

    if self.has_bias:
      new_layer.bias.data[:H_out] = self.module.bias.data.clone()
      new_layer.bias.data[H_out:] = self.module.bias.data[idx]

    self.module = new_layer

    # for batchnorm layer
    if self.has_bn:
      new_bn = nn.BatchNorm1d(H_out + H_new).to(device)
      new_bn.weight.data[:H_out] = self.bn.weight.data.clone()
      new_bn.weight.data[H_out:] = self.bn.weight.data[idx]
      new_bn.bias.data[:H_out] = self.bn.bias.data.clone()
      new_bn.bias.data[H_out:] = self.bn.bias.data[idx]
      new_bn.running_mean.data[:H_out] = self.bn.running_mean.data.clone()
      new_bn.running_mean.data[H_out:] = self.bn.running_mean.data[idx]
      new_bn.running_var.data[:H_out] = self.bn.running_var.data.clone()
      new_bn.running_var.data[H_out:] = self.bn.running_var.data[idx]
      self.bn = new_bn
    return H_new, idx

  def active_split(self, threshold, epsilon=1e-2):
    idx = np.argwhere(self.w <= threshold).reshape(
        -1)  # those are neurons ready for splitting
    H_new = len(idx)
    if H_new == 0:
      return 0, None

    H_out, H_in = self.module.weight.shape
    device = 'cuda' if self.module.weight.is_cuda else 'cpu'

    delta = torch.Tensor(self.v[idx, :]).to(device)
    idx = torch.LongTensor(idx).to(device)

    new_layer = nn.Linear(H_in, H_out + H_new, bias=self.has_bias).to(device)

    # for current layer
    new_layer.weight.data[:H_out, :] = self.module.weight.data.clone()
    new_layer.weight.data[H_out:, :] = self.module.weight.data[idx, :]
    new_layer.weight.data[idx, :] += epsilon * delta
    new_layer.weight.data[H_out:, :] -= epsilon * delta

    if self.has_bias:
      new_layer.bias.data[:H_out] = self.module.bias.data.clone()
      new_layer.bias.data[H_out:] = self.module.bias.data[idx]

    self.module = new_layer

    # for batchnorm layer
    if self.has_bn:
      new_bn = nn.BatchNorm1d(H_out + H_new).to(device)
      new_bn.weight.data[:H_out] = self.bn.weight.data.clone()
      new_bn.weight.data[H_out:] = self.bn.weight.data[idx]
      new_bn.bias.data[:H_out] = self.bn.bias.data.clone()
      new_bn.bias.data[H_out:] = self.bn.bias.data[idx]
      new_bn.running_mean.data[:H_out] = self.bn.running_mean.data.clone()
      new_bn.running_mean.data[H_out:] = self.bn.running_mean.data[idx]
      new_bn.running_var.data[:H_out] = self.bn.running_var.data.clone()
      new_bn.running_var.data[H_out:] = self.bn.running_var.data[idx]
      self.bn = new_bn
    return H_new, idx

  def passive_split(self, idx):
    H_new = idx.shape[0]
    H_out, H_in = self.module.weight.shape
    device = 'cuda' if self.module.weight.is_cuda else 'cpu'
    new_layer = nn.Linear(H_in + H_new, H_out, bias=self.has_bias).to(device)
    new_layer.weight.data[:, :H_in] = self.module.weight.data.clone()
    new_layer.weight.data[:, H_in:] = self.module.weight.data[:, idx] / 2.
    new_layer.weight.data[:, idx] /= 2.
    if self.has_bias:
      new_layer.bias.data = self.module.bias.data.clone()
    self.module = new_layer
