"""test cuda availability"""
import torch
print('__pyTorch VERSION:', torch.__version__)
print(torch.version.cuda)
print(torch.cuda.device_count())
print(torch.cuda.is_available())
# print ('Current cuda device ', torch.cuda.current_device())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

