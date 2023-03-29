import numpy as np
import scipy.io
from pa import power_amplifier_mp
import math
import torch
import torch.nn as nn

with open('nn_input.npy', 'rb') as f:
	nn_input = torch.from_numpy(np.load(f))
with open('nn_output.npy', 'rb') as f:
	real_output = np.load(f)
	nn_output = torch.from_numpy(real_output)

m = 2
k = 9
inputLayerDim = (2+k)*m
n_input, n_hidden, n_out, learning_rate = inputLayerDim, 40, 2, 0.001
model = nn.Sequential(nn.Linear(n_input, n_hidden),
					nn.ReLU(),
					nn.Linear(n_hidden, n_hidden),
					nn.ReLU(),
					nn.Linear(n_hidden, n_hidden),
					nn.ReLU(),
					nn.Linear(n_hidden, n_out)
					)
loss_function = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

losses = []
for epoch in range(100):
	pred_y = model(nn_input.float())
	loss = loss_function(pred_y, nn_output.float())
	losses.append(loss.item())
	model.zero_grad()
	loss.backward()
	optimizer.step()

import matplotlib.pyplot as plt
plt.plot(losses)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.title("Learning rate %f"%(learning_rate))
plt.show()

t = np.linspace(0,20,10500)
s = nn_output.float().detach().numpy()
p = pred_y.detach().numpy()


plt.plot(t, s)
plt.show()

plt.plot(t, p)
plt.show()

