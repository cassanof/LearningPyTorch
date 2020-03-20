import numpy as np
import torch

# IMPORTANT NOTE! Both grad variables and model needs to be on .cuda to make GPU work

# CPU
tensor_cpu = torch.rand(2, 2)

# CPU to GPU
if torch.cuda.is_available():
    tensor_cpu = tensor_cpu.cuda()
    tensor_gpu = torch.rand(2,2).cuda()  # How to create a tensor on gpu without conversion


print(tensor_cpu)
print(tensor_gpu)

# GPU to CPU
tensor_cpu = tensor_cpu.cpu()