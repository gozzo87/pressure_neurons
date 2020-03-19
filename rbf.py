class rbf(nn.Module):
  """the ground truth function of rbf.
  """

  def __init__(self):
    super(rbf, self).__init__()
    self.dh = 15
    self.theta = n n.Parameter(torch.randn(self.dh, 2) * 3)
    self.w = nn.Parameter(torch.randn(1, self.dh) * 3)

  def forward(self, x):
    """
      x : [B, 1]
      return : 1/15 w^T sigma(theta^T[x 1])
      """
    x = torch.cat((x, torch.ones_like(x)), -1)  # [B, 2]
    x = x.mm(self.theta.t())  # [B, 15]

    x = (-x.pow(2) / 2).exp()  # [B, 15]
    x = x.mm(self.w.t()) / self.dh  # [B, 1]
    return x


def generate_data():
  gt = rbf().to(device)
  x = torch.rand(1000, 1) * 10 - 5  # x ~ U(-5, 5)
  x = x.to(device)
  y = gt(x)
  x = x.cpu()
  y = y.cpu()
  data = [x, y, gt.state_dict()]
  torch.save(data, 'rbf.data')
  print('[INFO] succesfully generate 1000 data for rbf.')
