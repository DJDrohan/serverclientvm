import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(device)

import torch
print(torch.version.cuda)  # Should match your installed CUDA version
print(torch.cuda.is_available())  # Should return True if GPU is accessible
