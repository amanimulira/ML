# RNN - A class of neural networks that allow previous
# outputs to be used as inputs while having hidden states.

"""the core reason that RNNs are so exciting is that they
allow us to operate over sequences of vectors:

pros : inputs of any length -
model size not increasing with size of input
computation take into account historical information
weights are shared across time

cons : computation being slow
difficulty of accessing information from -
a long time ago
cannot consider any future input for the current state

"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from utils import All_LETTERS, N_LETTERS
from utlis import load_data, letter_to_tensor, line_to_tensor, random_training_example

class RNN(nn.Module):
    # nn.RNN
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input_tensor, hidden_tensor):
        combined = torch.cat((input_tensor, hidden_tensor), 1)

        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, self.hidden_size)

    category_lines, a




