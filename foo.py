import numpy as np

inputs = np.random.randn(32,8)
weights = np.random.randn(8, 10)
biases = np.random.randn(1, 10)
print(np.dot(inputs,weights))
print("\nB\n")
print(np.dot(inputs,weights) + biases)

print(np.ones((1,10)))