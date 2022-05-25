import numpy as np
import math
import matplotlib.pyplot as plt 
from operator import add
from scipy.fftpack import fft, ifft,fftshift
import cmath
PI = 3.14159265359

class power_amplifier_mp:
	def __init__(self, param, m,k):
		self.parameters = param
		self.K = k
		self.M = m
	
	def amplify(self,signal):
		X = self.calculateX(signal)
		return X.dot(self.parameters)
	
	def calculateX(self,signal):
		X = np.zeros((len(signal),self.M * self.K))
		for i in range(self.M,len(signal)):
			for j in range((self.K)*(self.M)):
					X[i][j] = signal[i-j%self.M] * abs(signal[i-j%self.M])**(math.floor(j/(self.M)))
		return X


class power_amplifier_gmp:
	def __init__(self, a,b,c):
		self.A = a
		self.B = b
		self.C = c
		
	def amplify(self,signal):
		output = signal*0
		for n in range(0,len(signal)):
			for k in range(0, self.A.shape[0]-1):
				for i in range(0,self.A.shaphe[1]-1):
					if(n-i >= 0):
						output[n] = output[n] + self.A[k][i] * ((abs(signal[n-i]))**k) * signal[n-i]
			for k in range(1, self.B.shape[0]):
				for i in range(0,self.B.shaphe[1]-1):
					for m in range(1,self.B.shaphe[2]):
						if(n-i >= 0 and n-i-m >= 0):
							output[n] = output[n] + self.B[k][i][m] * ((abs(signal[n-i-m]))**k) * signal[n-i]
			for k in range(1, self.C.shape[0]):
				for i in range(0,self.C.shaphe[1]-1):
					for m in range(1,self.C.shaphe[2]):
						if(n-i >= 0 and n-i+m >= 0):
							output[n] = output[n] + self.C[k][i][m] * ((abs(signal[n-i+m]))**k) * signal[n-i]
		return output
