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
# 
# # Random tensor
# tensor([[[0.9665, 0.6069, 0.6449, 0.5011, 0.0464],
#          [0.4076, 0.0796, 0.7424, 0.1142, 0.1473],
#          [0.3836, 0.9253, 0.8263, 0.5623, 0.2039],
#          [0.8924, 0.9407, 0.5542, 0.5013, 0.9184]],
# 
#         [[0.1796, 0.7418, 0.8621, 0.8446, 0.6124],
#          [0.5824, 0.5722, 0.0203, 0.8314, 0.7011],
#          [0.8154, 0.6490, 0.2502, 0.1764, 0.4401],
#          [0.7923, 0.4766, 0.4818, 0.3363, 0.5929]],
# 
#         [[0.3645, 0.3460, 0.0110, 0.4958, 0.1940],
#          [0.9041, 0.3930, 0.6580, 0.2478, 0.7775],
#          [0.4551, 0.0324, 0.7609, 0.4865, 0.5218],
#          [0.4687, 0.1446, 0.6360, 0.3054, 0.8408]]])
# Random tensor ndim -> 3
# Random tensor shape -> torch.Size([3, 4, 5])
# 
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
