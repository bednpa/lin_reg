# linear regression implementation with numpy

import numpy as np

# x
#  \
#   (*) -- y -- (-) -- a -- (^2) -- b -- (/N) -- loss
#  /           /
# w      y_real

# training data and labels
X = np.array([1,2,3,4], dtype=np.float32)
Y = np.array([5,10,15,20], dtype=np.float32)

# forward pass
def forward_pass(x, w):
    return w * x

# gradient computation
def backward_pass(x, y, y_real):
    # chain rule
    # 1/N * (w*x - y_real)^2 dloss/dw = 1/N * 2 * x (w*x - y_real)
    return np.dot(y - y_real, x * 2).mean()

# computing loss
def loss(y, y_real):
    # MSE
    return ((y - y_real)**2).mean()

# training
def train():
    w = 0.0
    learning_rate = 0.01
    i = 15
    test_number = 1.5

    for epoch in range(i):
        y = forward_pass(X, w)
        gradient = backward_pass(X, y, Y)
        w = w - learning_rate * gradient

        print(f'epoch {epoch + 1}, x = {test_number}, y = {forward_pass(test_number, w):.3f}, loss = {loss(y, Y):.3f}')

train()
