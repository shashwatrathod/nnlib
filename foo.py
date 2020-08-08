import numpy as np
import math
import timeit


inputs = np.random.randn(32,8)
# weights = np.random.randn(8, 10)
# biases = np.random.randn(1, 10)
# # # print(np.dot(inputs,weights))
# # print("\nB\n")
# # print(np.dot(inputs,weights) + biases)
#
# ret = np.maximum(0, [[-1,2,4,-6,0],[3,2,6,-1,-4]])
#
# print(ret)

x = [[2,3.0,4,5,-1,-2],[1,2,3,4,5,6],[2,3.0,4,5,-1,-2]]
z = [[2,3.0,4,5,-1,-2]]
x = np.array(x, dtype=float)
z = np.array(z,dtype=float)

a = np.exp(inputs)/np.reshape(np.sum(np.exp(inputs),axis=1),(-1,1))
print(a)

print(np.sum(a,axis=1))


