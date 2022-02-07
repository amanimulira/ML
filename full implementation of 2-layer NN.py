# Full implementation of training a 2 layer Neural Network
# Code taken from Lecture 4 - 88 ( Fei-Fei Li & justin Johnson & Serena Yeung)
import numpy as np
from numpy.random import randn

N, D_in, H, D_out = 64, 1000, 100, 10
x, y = randn(N, D_in), randn(N, D_out)
w1, w2 = randn(D_in, H), randn(H, D_out)

for t in range(2000):
    # sigmoid function (activation function)
    h = 1 / (1 + np.exp(-x.dot(w1)))
    y_pred = h.dot(w2)
    # loss function (HINGE LOSS ( SVM ))
    loss = np.square(y_pred - y).sum()
    print(t, loss)

    grad_y_pred = 2.0 * (y_pred - y)
    grad_w2 = h.T.dot(grad_y_pred)
    grad_h = grad_y_pred.dot(w2.T)
    # ∂h/∂w1 derivative of h.
    grad_w1 = x.T.dot(grad_h * h * (1 - h))
    # learning rate
    w1 -= 1e-4 * grad_w1
    w2 -= 1e-4 * grad_w2