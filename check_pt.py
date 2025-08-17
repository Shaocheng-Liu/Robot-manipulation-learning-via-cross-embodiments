import torch

checkpoint = torch.load("../logs/online_buffer_sawyer_peg-insert-side-v2/20000_39999.pt")
print(checkpoint[-1])