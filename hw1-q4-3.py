import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import math
from  scipy.integrate import quad

## Question 4.3
def guassian(x, mu, sigma):
	return stats.norm.pdf(x, mu, sigma)
	
mu1 = 0
variance1 = 1
sigma1 = math.sqrt(variance1)

mu2 = 1
variance2 = 2
sigma2 = math.sqrt(variance2)

x = np.linspace(mu2 - 3*sigma2, mu2 + 3*sigma2, 1000)
y1 = stats.norm.pdf(x, mu1, sigma1)
y2 = stats.norm.pdf(x, mu2, sigma2)
plt.plot(x, y1, label='P(x|L=1)')
plt.plot(x, y2, label='P(x|L=2)')

idx = np.argwhere(np.diff(np.sign(y1 - y2))).flatten()

p11 = quad(lambda x: guassian(x, mu1, sigma1), x[idx[0]], x[idx[1]])[0]
p22 = quad(lambda x: guassian(x, mu2, sigma2), -np.inf, x[idx[0]])[0] + quad(lambda x: guassian(x, mu2, sigma2), x[idx[1]], np.inf)[0]
p12 = quad(lambda x: guassian(x, mu2, sigma2), x[idx[0]], x[idx[1]])[0]
p21 = quad(lambda x: guassian(x, mu1, sigma1), -np.inf, x[idx[0]])[0] + quad(lambda x: guassian(x, mu1, sigma1), x[idx[1]], np.inf)[0]

print (p11, p22, p12, p21)
#p22 = getIntegrate([idx[0],idx[1]], y1)



