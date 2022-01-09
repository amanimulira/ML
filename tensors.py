import torch
a = torch.tensor([2., 3.], requires_grad=True)
b = torch.tensor([6.,4.], requires_grad=True)

"requires gad means that the computation will be tracked "

Q = 3*a**3 - b**2

"""
∂Q/∂a = 9a^2

∂Q/∂b = -2b
"""

external_grad = torch.tensor([1.,1.])
Q.backward(gradient=external_grad)

# check if collected gradients are correct
print(9*a**2 == a.grad)
print(-2*b == b.grad)

