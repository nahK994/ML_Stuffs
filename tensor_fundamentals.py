import torch
import numpy as sp
import pandas as pd
import matplotlib.pyplot as plt
print(torch.__version__)

#scalar
scalar = torch.tensor(7)
print("# Scalar")
print(scalar)
print(f"Scalar ndim -> {scalar.ndim}")
print(f"Scalar shape -> {scalar.shape}")
print(f"Scalar item -> {scalar.item()}")
print()

#vector
print("# Vector")
vector = torch.tensor([7, 7])
print(vector)
print(f"Vector ndim -> {vector.ndim}")
print(f"Vector shape -> {vector.shape}")
print()

#MATRIX
print("# Matrix")
matrix = torch.tensor([[7, 8, 9, 10], [11, 12, 13, 14], [15, 16, 17, 18]])
print(matrix)
print(f"Matrix ndim -> {matrix.ndim}")
print(f"Matrix shape -> {matrix.shape}")
print()

#TENSOR
print("# Tensor")
tensor = torch.tensor([
    [
      [7, 8, 9, 10],
      [11, 12, 13, 14],
      [15, 16, 17, 18]
    ],
    [
      [7, 8, 9, 10],
      [11, 12, 13, 14],
      [15, 16, 17, 18]
    ]
])
print(tensor)
print(f"Tensor ndim -> {tensor.ndim}")
print(f"Tensor shape -> {tensor.shape}")
print()

print("# Random tensor")
random_tensor = torch.rand(size=(3, 4, 5))
print(random_tensor)
print(f"Random tensor ndim -> {random_tensor.ndim}")
print(f"Random tensor shape -> {random_tensor.shape}")
print()

print("# All zeros")
zeros = torch.zeros(size=(2, 3, 5))
print(zeros)
print(f"zeros ndim -> {zeros.ndim}")
print(f"zeros shape -> {zeros.shape}")
print()

print("# Random ones")
ones = torch.ones(size=(1, 5, 2))
print(ones)
print(f"ones ndim -> {ones.ndim}")
print(f"ones shape -> {ones.shape}")
print()


# 2.4.0+cu121
# # Scalar
# tensor(7)
# Scalar ndim -> 0
# Scalar shape -> torch.Size([])
# Scalar item -> 7
# 
# # Vector
# tensor([7, 7])
# Vector ndim -> 1
# Vector shape -> torch.Size([2])
# 
# # Matrix
# tensor([[ 7,  8,  9, 10],
#         [11, 12, 13, 14],
#         [15, 16, 17, 18]])
# Matrix ndim -> 2
# Matrix shape -> torch.Size([3, 4])
# 
# # Tensor
# tensor([[[ 7,  8,  9, 10],
#          [11, 12, 13, 14],
#          [15, 16, 17, 18]],
# 
#         [[ 7,  8,  9, 10],
#          [11, 12, 13, 14],
#          [15, 16, 17, 18]]])
# Tensor ndim -> 3
# Tensor shape -> torch.Size([2, 3, 4])
# 
# # Random tensor
# tensor([[[0.9593, 0.5790, 0.5479, 0.3142, 0.4065],
#          [0.3809, 0.5403, 0.2724, 0.8962, 0.6887],
#          [0.8041, 0.1434, 0.2294, 0.6005, 0.3660],
#          [0.5448, 0.4092, 0.4644, 0.7976, 0.7361]],
# 
#         [[0.1541, 0.3451, 0.3813, 0.7162, 0.1565],
#          [0.3663, 0.7831, 0.2783, 0.3772, 0.9370],
#          [0.5630, 0.4592, 0.8073, 0.8256, 0.6226],
#          [0.1359, 0.5936, 0.5800, 0.5049, 0.6013]],
# 
#         [[0.9686, 0.0877, 0.1168, 0.2142, 0.7557],
#          [0.4575, 0.3597, 0.9019, 0.6966, 0.7327],
#          [0.4631, 0.7497, 0.7978, 0.3195, 0.9291],
#          [0.4526, 0.0896, 0.6439, 0.2300, 0.7181]]])
# Random tensor ndim -> 3
# Random tensor shape -> torch.Size([3, 4, 5])
# 
# # All zeros
# tensor([[[0., 0., 0., 0., 0.],
#          [0., 0., 0., 0., 0.],
#          [0., 0., 0., 0., 0.]],
# 
#         [[0., 0., 0., 0., 0.],
#          [0., 0., 0., 0., 0.],
#          [0., 0., 0., 0., 0.]]])
# zeros ndim -> 3
# zeros shape -> torch.Size([2, 3, 5])
# 
# # Random ones
# tensor([[[1., 1.],
#          [1., 1.],
#          [1., 1.],
#          [1., 1.],
#          [1., 1.]]])
# ones ndim -> 3
# ones shape -> torch.Size([1, 5, 2])
# 
# 
