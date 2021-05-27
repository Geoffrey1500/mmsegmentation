from mmseg.models import ResNet
import torch

self = ResNet(depth=18, in_channels=6)
self.eval()
inputs = torch.rand(1, 6, 32, 32)
level_outputs = self.forward(inputs)
for level_out in level_outputs:
    print(tuple(level_out.shape))

# import torch
# flag = torch.cuda.is_available()
# if flag:
#     print("CUDA可使用")
# else:
#     print("CUDA不可用")
#
# ngpu= 1
# # Decide which device we want to run on
# device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
# print("驱动为：",device)
# print("GPU型号： ",torch.cuda.get_device_name(0))
