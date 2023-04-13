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
	print("Calculating...")
	input = scipy.io.loadmat('../pa_data/input.mat')
	output = scipy.io.loadmat('../pa_data/output.mat')
	input = input['input']
	output = output['output']
	signal = input[0]
	errMap = np.zeros((10,10))
	min_err = 100
	min_m = 0
	min_k = 0
	for m in range(1,11):
		for k in range(1,11):
			pa1 = power_amplifier_mp([], m,k)
			X = pa1.calculateX(signal)
			ThetaLS = (np.matmul(inv(np.matmul((np.transpose(X)).conjugate(),X)),(np.transpose(X)).conjugate())).dot(np.transpose(output))
			pa2 = power_amplifier_mp(ThetaLS,m,k)
			amp_signal = pa2.amplify(signal)
			err = (np.linalg.norm(abs(np.transpose(output)) - abs(amp_signal)))/np.linalg.norm(abs(np.transpose(output)))
			if(err < min_err):
				min_m = m
				min_k = k
				min_err = err
			errMap[m-1][k-1] = 20*math.log10(err)
	print("minimum_total:")
	print("m: " + str(min_m) + ", k: " + str(min_k)+ ", err: " + str(20*math.log10(min_err)))
	fig1, (ax1)= plt.subplots(1, sharex = True, sharey = False)
	ax1.imshow(errMap, interpolation ='none', aspect = 'auto')
	for (j,i),label in np.ndenumerate(errMap):
		label = "{:.2f}".format(label)
		ax1.text(i,j,label,ha='center',va='center',color ="white")
	plt.xlabel('k-1')
	plt.ylabel('m-1')
	plt.title("Error in db")
	plt.show()

main()