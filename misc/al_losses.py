def gaussian_log_loss(output, target, logvar):
  term1 = 0.5*torch.exp(-logvar)*(target - output)**2
  term2 = 0.5*logvar
  return (term1 + term2).sum()

def laplacian_log_loss(output, target, logvar):
  term1 = 0.5*torch.exp(-logvar)*torch.abs(target - output)
  term2 = 0.5*logvar
  return (term1 + term2).sum()