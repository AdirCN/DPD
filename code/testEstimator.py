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
	AMP = [5,5,5,5]
	FREQ = [3,-3,5,-5]
	PHASE =  [0,0,0,0]
	ONES = [1,1,1,1]
	Fs = 10000
	time = 100
	signal1 = signal_generator(AMP,FREQ,PHASE,time, Fs)
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
	pa = power_amplifier_mp(ThetaLS,m,k)
	amp_signal = pa.amplify(signal1.signal)
	
	X2 = pa1.calculateX(signal1.signal)
	ThetaLS2 = (np.matmul(inv(np.matmul((np.transpose(X2)).conjugate(),X2)),(np.transpose(X2)).conjugate())).dot(amp_signal)
	print(20*np.log10(np.linalg.norm(abs(ThetaLS) - abs(ThetaLS2))/np.linalg.norm(abs(ThetaLS))))

	
main()
