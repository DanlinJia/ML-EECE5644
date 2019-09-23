import matplotlib.pyplot as plt
import numpy as np


a1 = 0
b1 = 1
a2 =1
b2 = 2 

x = np.linspace(0, 5, 100)
y = abs(a2 - x)/b2 - abs(a1-x)/b1

plt.plot(x, y, label='linear')

plt.xlabel('x')
plt.ylabel('l(x)')

plt.title("Log-likelihood-ratio Function")

#plt.legend()

plt.show()