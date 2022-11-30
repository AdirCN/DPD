import numpy as np
import scipy.io
from pa import power_amplifier_mp
from torch import nn
import torch

input = scipy.io.loadmat('../pa_data/input.mat')
output = scipy.io.loadmat('../pa_data/output.mat')
input = input['input']
output = output['output']
input = input[0]
output = output[0]

m=2
k=9
lin=50*m*k
lout = 50
pa1 = power_amplifier_mp([], m,k)
X = pa1.calculateX(input)
X = X.flatten('C')
X = [a for b in zip(X.real, X.imag) for a in b]
X = np.array(X)
X = np.pad(X, (0, lin-len(X)%lin), 'constant', constant_values=(0, 0))
X = np.transpose(X.reshape((lin, int(len(X)/lin))))
X = torch.from_numpy(np.float32(X))
output = [a for b in zip(output.real, output.imag) for a in b]
output = np.array(output)
output = np.pad(output, (0, lout-len(output)%lout), 'constant', constant_values=(0, 0))
output = np.transpose(output.reshape((lout, int(len(output)/lout))))
output = torch.from_numpy(np.float32(output))

rho = 1
min = -1
min_index = 0

while(rho <= 100):
	model = nn.Sequential(nn.Linear(lin, rho),
						  nn.ReLU(),
						  nn.Linear(rho, lout))
						  
	# Define the loss
	criterion = torch.nn.MSELoss(reduction='sum')

	# Optimizers require the parameters to optimize and a learning rate
	optimizer = torch.optim.SGD(model.parameters(), lr=0.0000000000000000000000001)
	epochs = 5
	for e in range(epochs):
		running_loss = 0
		optimizer.zero_grad()
		YHat = model(X)
		loss = criterion(YHat, output)
		loss.backward()
		optimizer.step()  
		running_loss += loss.item()
		outputNorm = np.linalg.norm(output.detach().numpy())
		lossNorm = np.linalg.norm(YHat.detach().numpy() - output.detach().numpy())
		print(lossNorm**2 / outputNorm**2)
		
	print(f"Training loss: {running_loss}, rho: {rho}")
	if(min == -1):
		min = running_loss
		yhat_min = YHat
	else:
		if(running_loss <= min):
			min = running_loss
			min_index = rho
			yhat_min = YHat
	rho = rho+1

print(min_index)
print(min)
print(yhat_min)
print(output)
print(yhat_min.shape)