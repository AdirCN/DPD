import numpy as np
import matplotlib.pyplot as plt 

s = np.load("errMap.npy")
s = 20*np.log10(s)

fig1, (ax1)= plt.subplots(1, sharex = True, sharey = False)
ax1.imshow(s, interpolation ='none', aspect = 'auto')
for (j,i),label in np.ndenumerate(s):
	label = "{:.2f}".format(label)
	ax1.text(i,j,label,ha='center',va='center')

plt.xlabel('k-1')
plt.ylabel('m-1')
plt.title("Error in db")
plt.show()
