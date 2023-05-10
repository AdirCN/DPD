from pa import power_amplifier_mp
from pa import power_amplifier_gmp
from sg import signal_generator
import numpy as np
import math
import matplotlib.pyplot as plt 
from operator import add
from scipy.fftpack import fft, ifft,fftshift
import cmath
import scipy.io
from numpy.linalg import inv
PI = 3.14159265359

def main():
	m = 2
	k = 9
	input = scipy.io.loadmat('../pa_data/input.mat')
	output = scipy.io.loadmat('../pa_data/output.mat')
	input = input['input']
	output = output['output']
	pa1 = power_amplifier_mp([], m,k)
	X = pa1.calculateX(input[0])
	ThetaLS = (np.matmul(inv(np.matmul((np.transpose(X)).conjugate(),X)),(np.transpose(X)).conjugate())).dot(np.transpose(output))

	ThetaGD = ThetaLS*0
	mu = 0.005*inv(np.matmul(np.transpose(X).conjugate(),X) + 0.00000000000001)
	err = []
	for i in range(1,1000):
		err.append(np.linalg.norm(np.matmul(X,ThetaGD) - np.transpose(output))/np.linalg.norm(np.transpose(output)))
		ThetaGD = ThetaGD - 2*np.matmul(mu,np.matmul(np.transpose(X).conjugate(),np.matmul(X,ThetaGD) - np.transpose(output)))
	print(np.linalg.norm(ThetaLS-ThetaGD)/np.linalg.norm(ThetaLS))
	
	plt.plot(range(1,1000),err)
	plt.title('Normalized error between SGD modeled PA output and real Labs output')
	plt.xlabel('Iteration number')
	plt.ylabel('Normalized error')
	plt.draw() 
	plt.show()
	
main()