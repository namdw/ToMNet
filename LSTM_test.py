#!/usr/bin/env python3

import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from CLSTM import LSTM, CLSTM

print('hi')

batch_size = 1
input_size = 1
output_size = 1

model = CLSTM(batch_size, input_size, output_size)

input = Variable(torch.randn(batch_size, 3, input_size))
print(input.size())
print()

output = model(input)


criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

for i in range(2000):
	random_i = np.random.randint(0,9)
	data = 0.1*np.array([[random_i, random_i+1, random_i+2]])
	y = 0.1*np.array([[random_i+3]])
	data = torch.Tensor(data)
	# data.unsqueeze(0)
	data = data.unsqueeze(2)
	optimizer.zero_grad()
	output = model(Variable(data))
	loss = criterion(output, Variable(torch.Tensor(y)))
	loss.backward()
	optimizer.step()

test_data = Variable(torch.Tensor(0.1*np.array([[2,3]])))
test_data = test_data.unsqueeze(2)

print(model(test_data))

# print(input, output)