from mindspore import Tensor
#import torch.nn as nn
import mindspore
import mindspore.nn as nn
import numpy as np
import mindspore.ops as ops

# x = np.array([1, 2, 3])
# print(x)
# x = Tensor(x, mindspore.float64)
# print(x)
# x = Tensor.sqrt(x)
# print(x)
# print(x.dtype)
# y = np.array([5, 6, 7])
# print(y)

# expand_dims = mindspore.ops.ExpandDims()
# x = expand_dims(x, 0)
# x = expand_dims(x, 0)
# print(x)

# x = Tensor(np.array([[1.0, 4.0, 9.0]]), mindspore.float32)
# sqrt = ops.Sqrt()
# output = sqrt(x)
# output2 = Tensor.sqrt(x)
# print(output)
# print(output2)

# z = ops.ones((3, 4),mindspore.float32)
# print(z)

angle = Tensor([[1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]])
if(angle>5):
    angle = 0;
print(angle)