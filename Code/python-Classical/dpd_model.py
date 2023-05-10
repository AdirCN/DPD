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
from sklearn.linear_model import LinearRegression
PI = 3.14159265359

def main():
	input = scipy.io.loadmat('../pa_data/input.mat')
	output = scipy.io.loadmat('../pa_data/output.mat')
	input = input['input']
	output = output['output']
	signal = input[0]

	m=2
	k=9
	G = 0.01708039
	
	pa1 = power_amplifier_mp([], m,k)
	scale_factor = max(signal.flatten(),key=abs)/(max(output.flatten(),key=abs))
	scaled_output = output.flatten()*scale_factor
	Y = pa1.calculateX(scaled_output)
	ThetaPDLS = (np.matmul(inv(np.matmul((np.transpose(Y)).conjugate(),Y)),(np.transpose(Y)).conjugate())).dot(np.transpose(signal))
	
	pa1 = power_amplifier_mp([], m,k)
	X = pa1.calculateX(input[0])
	ThetaLS = (np.matmul(inv(np.matmul((np.transpose(X)).conjugate(),X)),(np.transpose(X)).conjugate())).dot(np.transpose(output))
	
	pa2 = power_amplifier_mp(ThetaLS,m,k)
	dpd2 = power_amplifier_mp(ThetaPDLS,m,k)
	
	amp_signal_with_dpd = pa2.amplify(dpd2.amplify(signal))
	amp_signal_without_dpd = pa2.amplify(signal)
	
	plt.plot(abs(signal),abs(amp_signal_without_dpd.flatten()),'.', markersize=1)
	plt.plot(abs(signal),abs(output.flatten()),'.', markersize=1)
	plt.title('Modeled and measured AM_AM')
	plt.xlabel('Amplitude')
	plt.ylabel('Amplitude')
	plt.draw() 
	plt.show()
	
	plt.plot(abs(signal),abs(amp_signal_without_dpd.flatten()),'.', markersize=1)
	plt.plot(abs(signal),abs(amp_signal_with_dpd.flatten()),'.', markersize=1)
	plt.title('AM_AM with and without dpd')
	plt.xlabel('Amplitude')
	plt.ylabel('Amplitude')
	plt.draw() 
	plt.show()
	
	x = np.asarray(abs(signal))
	y = np.asarray(abs(amp_signal_with_dpd.flatten()))
	x = x.reshape((-1, 1))
	# Create an instance of a linear regression model and fit it to the data with the fit() function:
	model = LinearRegression().fit(x, y) 
	# The following section will get results by interpreting the created instance: 
	# Obtain the coefficient of determination by calling the model with the score() function, then print the coefficient:
	r_sq = model.score(x, y)
	slope = model.coef_
	desired = np.transpose(slope*abs(input))
	err = 10*math.log10((np.linalg.norm(abs(desired)-abs(amp_signal_with_dpd))/np.linalg.norm(desired)))
	
	e = 0.00000000000001
	a = np.arctan(np.divide(signal.imag,signal.real + e))
	b = np.arctan(np.divide(amp_signal_without_dpd.flatten().imag,amp_signal_without_dpd.flatten().real+e))
	c = np.arctan(np.divide(output.flatten().imag,output.flatten().real+e))
	plt.plot(abs(signal),np.transpose(a-np.transpose(b)),'.', markersize=1)
	plt.plot(abs(signal), np.transpose(a - np.transpose(c)),'.', markersize=1)
	plt.title('Modeled and measured AM_PM')
	plt.xlabel('Amplitude')
	plt.ylabel('phase difference')
	plt.draw() 
	plt.show()
	
	
	
	e = 0.00000000000001
	a = np.arctan(np.divide(signal.imag,signal.real + e))
	b = np.arctan(np.divide(amp_signal_without_dpd.flatten().imag,amp_signal_without_dpd.flatten().real+e))
	c = np.arctan(np.divide(amp_signal_with_dpd.flatten().imag,amp_signal_with_dpd.flatten().real+e))
	plt.plot(abs(signal),np.transpose(a-np.transpose(b)),'.', markersize=1)
	plt.plot(abs(signal), np.transpose(a - np.transpose(c)),'.', markersize=1)
	plt.title('AM_PM with and without dpd')
	plt.xlabel('Amplitude')
	plt.ylabel('phase difference')
	plt.draw() 
	plt.show()
	
	rate = 3*(10**6)
	f = np.linspace(-rate/2,rate/2,len(amp_signal_without_dpd.flatten()))
	amp_spectrum_without_dpd = fftshift(fft(amp_signal_without_dpd.flatten()))
	amp_spectrum_output = fftshift(fft(output.flatten()))
	
	plt.plot(f,10*np.log10(np.abs(amp_spectrum_without_dpd)),f,10*np.log10(np.abs(amp_spectrum_output)))
	plt.title('Modeled and measured output spectrum')
	plt.xlabel('f [Hz]')
	plt.ylabel('Amplitude [dB]')
	plt.draw() 
	plt.show()
	
	
	rate = 3*(10**6)
	f = np.linspace(-rate/2,rate/2,len(amp_signal_without_dpd.flatten()))
	amp_spectrum_without_dpd = fftshift(fft(amp_signal_without_dpd.flatten()))
	amp_spectrum_with_dpd = fftshift(fft(amp_signal_with_dpd.flatten()))
	
	plt.plot(f,10*np.log10(np.abs(amp_spectrum_without_dpd)),f,10*np.log10(np.abs(amp_spectrum_with_dpd)))
	plt.title('Output spectrum with and without dpd')
	plt.xlabel('f [Hz]')
	plt.ylabel('Amplitude [dB]')
	plt.draw() 
	plt.show()
	
	oobwidpd = (np.linalg.norm(np.square(abs(amp_spectrum_with_dpd[1100:4500])))+np.linalg.norm(np.square(abs(amp_spectrum_with_dpd[6000:9400]))))/np.linalg.norm(np.square(abs(amp_spectrum_with_dpd[4500:6000])));
	oobwodpd = (np.linalg.norm(np.square(abs(amp_spectrum_without_dpd[1100:4500])))+np.linalg.norm(np.square(abs(amp_spectrum_without_dpd[6000:9400]))))/np.linalg.norm(np.square(abs(amp_spectrum_without_dpd[4500:6000])));
	OOB = 10*math.log10(oobwidpd/oobwodpd);
	
main()