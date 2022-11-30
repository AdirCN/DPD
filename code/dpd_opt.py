
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
	input = scipy.io.loadmat('../pa_data/input.mat')
	output = scipy.io.loadmat('../pa_data/output.mat')
	input = input['input']
	output = output['output']
	signal = input[0]
	t = np.linspace(0,20,10500)
	errMap = np.zeros((10,10))
	min_err = 100
	min_m = 0
	min_k = 0
	for m in range(0,10):
		for k in range(0,10):
			G = 0.01708039
			pa1 = power_amplifier_mp([], m,k)
			scale_factor = max(signal)/(G*max(output.flatten()))
			scaled_output = output.flatten()*scale_factor
			Y = pa1.calculateX(scaled_output)
			ThetaLS = (np.matmul(inv(np.matmul((np.transpose(Y)).conjugate(),Y)),(np.transpose(Y)).conjugate())).dot(np.transpose(signal))
			x_hat = Y.dot(ThetaLS)
			pa2 = power_amplifier_mp(ThetaLS,m,k)
			amp_signal = pa2.amplify(scaled_output)
			err = (np.linalg.norm(abs(np.transpose(signal)) - abs(amp_signal)))/np.linalg.norm(abs(signal))
			if(err < min_err):
				min_m = m
				min_k = k
				min_err = err
			print("m: " + str(m) + ", k: " + str(k)+ ", err: " + str(20*math.log(err)))
	print("minimum_total:")
	print("m: " + str(min_m) + ", k: " + str(min_k)+ ", err: " + str(20*math.log(min_err)))
	'''
	plt.plot(t,np.transpose(abs(input)))
	plt.title('input signal')
	plt.xlabel('t [sec]')
	plt.ylabel('Amplitude')
	plt.draw() 
	plt.show()
	
	plt.plot(t,abs(output.flatten()))
	plt.title('Amplified')
	plt.xlabel('t [sec]')
	plt.ylabel('Amplitude')
	plt.draw() 
	plt.show()
	
	plt.plot(abs(signal),abs(amp_signal),'.', markersize=1)
	plt.title('AM_AM')
	plt.xlabel('Amplitude')
	plt.ylabel('Amplitude')
	plt.draw() 
	plt.show()
	e = 0.0000000000000000000000001
	a = np.arctan(np.divide(signal.imag,signal.real + e))
	b = np.arctan(np.divide(amp_signal.imag,amp_signal.real+e))
	plt.plot(abs(signal), np.transpose(a - np.transpose(b)),'.', markersize=1)
	plt.title('AM_PM')
	plt.xlabel('Amplitude')
	plt.ylabel('phase difference')
	plt.draw() 
	plt.show()
	
	rate = 3*(10**6)
	f = np.linspace(-rate/2,rate/2,len(amp_signal))
	amp_spectrum = fftshift(fft(amp_signal.flatten()))
	plt.plot(f,10*np.log(np.abs(amp_spectrum)))
	plt.title('Modeled PA output Spectrum')
	plt.xlabel('f [Hz]')
	plt.ylabel('Amplitude [dB]')
	plt.draw() 
	plt.show()
		
	amp_spectrum = fftshift(fft(signal))
	plt.plot(f,10*np.log(np.abs(amp_spectrum)))
	plt.title('PA input Spectrum')
	plt.xlabel('f [Hz]')
	plt.ylabel('Amplitude [dB]')
	plt.draw() 
	plt.show()
	'''
	
main()