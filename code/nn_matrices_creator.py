import numpy as np
import scipy.io
from pa import power_amplifier_mp
import math

# Extract data
input = scipy.io.loadmat('../pa_data/input.mat')
output = scipy.io.loadmat('../pa_data/output.mat')
input = input['input']
output = output['output']
input = input[0]
output = output[0]

m = 2
k = 9
inputLayerDim = 2*m +k*m
nnInputMat = np.zeros((len(input),inputLayerDim))

for i in range(1,len(output)):
	for j in range(m):
		nnInputMat[i][j] = input.real[i-j]
		nnInputMat[i][j+m] = input.imag[i-j]
	for j in range(2*m,(2+k)*m):
		power = (math.floor(j/m) - 1)
		nnInputMat[i][j] = abs(input[i-j%m])**power
		
with open('nn_input.npy', 'wb') as f:
	np.save(f, nnInputMat)
	
nnOutputMat = np.zeros((len(output),2))
for i in range(1,len(output)):
	nnOutputMat[i][0] = output.real[i]
	nnOutputMat[i][1] = output.imag[i]

with open('nn_output.npy', 'wb') as f:
	np.save(f, nnOutputMat)
