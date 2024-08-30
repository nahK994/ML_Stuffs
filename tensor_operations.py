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

# Shapes need to be in the right way
tensor_A = torch.tensor([[1, 2],
                         [3, 4],
                         [5, 6]], dtype=torch.float32)

tensor_B = torch.tensor([[7, 10],
                         [8, 11],
                         [9, 12]], dtype=torch.float32)

print(f"tensor A ==> {tensor_A}")
print(f"tensor B.T ==> {tensor_B.T}")
print(f"matmul ==> {torch.matmul(tensor_A, tensor_B.T)}")



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
# tensor([[[0.6011, 0.5786, 0.2975, 0.3133, 0.2302],
#          [0.3642, 0.5583, 0.7535, 0.0178, 0.4767],
#          [0.4270, 0.1643, 0.5188, 0.0689, 0.7314],
#          [0.5112, 0.2988, 0.3777, 0.4498, 0.7286]],
# 
#         [[0.9567, 0.3037, 0.8276, 0.6125, 0.6860],
#          [0.7086, 0.7636, 0.1166, 0.1868, 0.5479],
#          [0.5567, 0.3997, 0.7943, 0.8171, 0.0299],
#          [0.2326, 0.1450, 0.1957, 0.6422, 0.4927]],
# 
#         [[0.8115, 0.3238, 0.9201, 0.6609, 0.4619],
#          [0.3043, 0.0294, 0.5179, 0.0462, 0.6052],
#          [0.8212, 0.2676, 0.7335, 0.5599, 0.3615],
#          [0.8713, 0.5516, 0.9494, 0.0555, 0.1213]]])
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
# Original shapes: tensor_A = torch.Size([3, 2]), tensor_B = torch.Size([3, 2])
# 
# New shapes: tensor_A = torch.Size([3, 2]) (same as above), tensor_B.T = torch.Size([2, 3])
# 
# Multiplying: torch.Size([3, 2]) * torch.Size([2, 3]) <- inner dimensions match
# 
# Output:
# 
# tensor([[ 27.,  30.,  33.],
#         [ 61.,  68.,  75.],
#         [ 95., 106., 117.]])
# 
# Output shape: torch.Size([3, 3])
# 
