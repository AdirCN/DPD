import numpy as np
import math
import matplotlib.pyplot as plt 
from operator import add
from scipy.fftpack import fft, ifft,fftshift
import cmath

PI = 3.14159265359

def myExp(amp,freq,phase,t):
		return amp*cmath.exp(1j*(2*PI*freq*t + phase))

'''
 This class represents one signal. The signal consists of a superposition of independent complex exponents.  
 initialization input:
	amp: vector of amplitudes for complex exponents.
	freq: vector of frequencies (Hz) for complex exponents.
	phase: vector of phases for complex exponents.
	duration: duration of signal in seconds.
	rate: sample rate	
'''
class signal_generator:
	def __init__(self, amp, freq, phase, duration,rate):
		self.amplitudes = amp
		self.frequencies = freq
		self.phases = phase
		self.signal = []
		self.spectrum = []
		self.t = np.linspace(0,duration,duration*rate)
		self.f = np.linspace(-rate/2,rate/2,duration*rate)
		
	def generate(self):
		vExp = np.vectorize(myExp)
		for i in range(0,len(self.amplitudes)):
			temp = vExp(self.amplitudes[i],self.frequencies[i],self.phases[i],self.t)
			if(np.array(self.signal).size == 0):
				self.signal = temp*0
			self.signal = list(map(add,self.signal, temp))
			self.signal = np.array(self.signal)
			
	def plot_spectrum(self):
		self.spectrum = fftshift(fft(self.signal))
		plt.plot(self.f,np.abs(self.spectrum))
		plt.title('Spectrum')
		plt.xlabel('f [Hz]')
		plt.ylabel('Amplitude')
		plt.draw() 
		plt.show()
	
	def plot_time_variation(self):
		plt.plot(self.t,abs(self.signal))
		plt.title('Time variation')
		plt.xlabel('t [sec]')
		plt.ylabel('Amplitude')
		plt.draw() 
		plt.show()
	
	def PAPR(self):
		peak_amp = max(np.absolute(self.signal))
		rms = np.sqrt(np.mean(np.absolute(self.signal)**2))
		papr = (peak_amp/rms)**2
		return papr
		
	def PAPR_DB(self):
		peak_amp = max(np.absolute(self.signal))
		rms = np.sqrt(np.mean(np.absolute(self.signal)**2))
		papr = (peak_amp/rms)**2
		return 20*np.log10(papr)
	
	def Power_histogram(self,b):
		power = np.absolute(self.signal)**2;
		plt.hist(power, bins = b)
		plt.title('Power histogram')
		plt.xlabel('Power')
		plt.ylabel('Number of occurrences')
		plt.show()
		

'''
# Example of signal generation 
def main():
	AMP = [1,2,1,2,3,2,1,2,3,2,1,2,3,2,1,2,1,2,1,2]
	FREQ = [12,3,2,1,2,45,43,67,87,56,48,73,29,87,10,76,23,35,45,76]
	PHASE =  [0,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1]
	ONES = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
	signal1 = signal_generator(AMP,FREQ,PHASE,max(np.divide(ONES,[abs(element) for element in FREQ])), 20000)
	signal1.generate()
	signal1.plot_spectrum()
	signal1.plot_time_variation()
	print("PAPR: " + str(signal1.PAPR()) + " = " + str(signal1.PAPR_DB()) + " db")
	signal1.Power_histogram(100)
	plt.close()
	return 0
main()
'''

