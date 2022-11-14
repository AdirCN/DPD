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
	AMP = [1,5,7,3,3,4,1]
	FREQ = [3,7,6,3,2,-4,-9]
	PHASE =  [1,4,3,6,5,4,6]
	ONES = [1,1,1,1,1,1,1]
	signal1 = signal_generator(AMP,FREQ,PHASE,5, 1000)
	signal1.generate()
	
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
	print(mu.shape)
	for i in range(1,1000):
		ThetaGD = ThetaGD - 2*np.matmul(mu,np.matmul(np.transpose(X).conjugate(),np.matmul(X,ThetaGD) - np.transpose(output)))
	print(np.linalg.norm(ThetaLS-ThetaGD)/np.linalg.norm(ThetaLS))
	'''
	#LS
	signal1.plot_time_variation()
	pa = power_amplifier_mp(ThetaLS,m,k)
	amp_signal = pa.amplify(signal1.signal)
	
	plt.plot(signal1.t,abs(amp_signal))
	plt.title('Amplified')
	plt.xlabel('t [sec]')
	plt.ylabel('Amplitude')
	plt.draw() 
	plt.show()
	
	plt.plot(abs(signal1.signal),abs(amp_signal),'.', markersize=1)
	plt.title('AM_AM')
	plt.xlabel('Amplitude')
	plt.ylabel('Amplitude')
	plt.draw() 
	plt.show()
	e = 0.0000000000000000000000001
	a = np.arctan(np.divide(signal1.signal.imag,signal1.signal.real+e))
	b = np.arctan(np.divide(amp_signal.imag,amp_signal.real+e))
	plt.plot(abs(signal1.signal), np.transpose(a - np.transpose(b)),'.', markersize=1)
	plt.title('AM_PM')
	plt.xlabel('Amplitude')
	plt.ylabel('Phase difference')
	plt.draw() 
	plt.show()
	'''
main()