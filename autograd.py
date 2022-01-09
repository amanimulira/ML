#back propagation

import torch
x = torch.ones(5)
y = torch.zeros(3)
w = torch.randn(5, 3, requires_grad=True)
b = torch.randn(3, requires_grad=True)
z = torch.matmul(x, w)+b

loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)

print("Gradient function for z =", z.grad_fn)
print("Gradient funtion for loss =", loss.grad_fn)

loss.backward()

print(w.grad)
print(b.grad)

"""
In a forward pass, autograd does two things simultaneously:

run the requested operation to compute a resulting tensor

maintain the operation’s gradient function in the DAG.

The backward pass kicks off when .backward() is called on the DAG root. autograd then:

computes the gradients from each .grad_fn,
accumulates them in the respective tensor’s .grad attribute
using the chain rule, propagates all the way to the leaf tensors.
"""


inp = torch.eye(5, requires_grad=True)
out = (inp+1).pow(2)
out.backward(torch.ones_like(inp), retain_graph=True)
print("First call \n",inp.grad)
out.backward(torch.ones_like(inp), retain_graph=True)
print("\n Second call \n", inp.grad)
inp.grad.zero_()
out.backward(torch.ones_like(inp), retain_graph=True)
print("\n Third call \n", inp.grad)