import torch
import torch.nn.functional as F
h_hat = torch.load("./comcloak/training/save_tensors/h_hat_step15970.pt")
h_ri = torch.load("./comcloak/training/save_tensors/h_ri_step15970.pt")
mean = h_ri.mean(dim=[0,1,2,3], keepdim=True)
std = h_ri.std(dim=[0,1,2,3], keepdim=True)
H_norm = (h_ri - mean) / (std + 1e-6)
H_hat_norm = (h_hat - mean) / (std + 1e-6)
# denom = torch.mean(torch.abs(h_ri)**2) + 1e-8
# loss = torch.mean(torch.abs(h_ri - h_hat)**2) / denom
loss = F.mse_loss(H_hat_norm, H_norm)
print(loss.item())
