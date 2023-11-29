import torch

def RBF(inputs: torch.Tensor, r_cut: float,output_size: int=20):
    n = torch.arange(1,output_size+1).to(inputs.device)
    return ((torch.sin((n * torch.pi / r_cut) * inputs)) / inputs)
   
def fcut(inputs: torch.Tensor, r_cut: float):
    f_c = 0.5 * (torch.cos(torch.pi * inputs / r_cut) + 1) * (inputs<r_cut).float()
    return f_c

def mse(preds: torch.Tensor, targets: torch.Tensor):
    return torch.mean((preds - targets).square())

def mae(preds: torch.Tensor, targets: torch.Tensor):
    return torch.mean(torch.abs(preds - targets))