from pa import power_amplifier_mp
from pa import power_amplifier_gmp
from sg import signal_generator
import numpy as np
import math
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
	t = np.linspace(0,20,10500)
	errMap = np.zeros((10,10))
	m=2
	k=9
	pa1 = power_amplifier_mp([], m,k)
	X = pa1.calculateX(input[0])
	ThetaLS = (np.matmul(inv(np.matmul((np.transpose(X)).conjugate(),X)),(np.transpose(X)).conjugate())).dot(np.transpose(output))
	pa2 = power_amplifier_mp(ThetaLS,m,k)
	amp_signal = pa2.amplify(signal)	
	x = []
	y = []
	
	for i in range(0,len(abs(signal))):
		if(abs(signal)[i] < 5):
			x.append(abs(signal)[i])
			y.append(abs(amp_signal)[i])
	
	x = np.asarray(x)
	y = np.asarray(y)
	# Import the packages and classes needed in this example:
	x = x.reshape((-1, 1))

	# Create an instance of a linear regression model and fit it to the data with the fit() function:
	model = LinearRegression().fit(x, y) 

	# The following section will get results by interpreting the created instance: 

	# Obtain the coefficient of determination by calling the model with the score() function, then print the coefficient:
	r_sq = model.score(x, y)
	print('coefficient of determination:', r_sq)

	# Print the Intercept:
	print('intercept:', model.intercept_)

	# Print the Slope:
	print('slope:', model.coef_) 

	# Predict a Response and print it:
	y_pred = model.predict(x)
	print('Predicted response:', y_pred, sep='\n')
	
	
main()