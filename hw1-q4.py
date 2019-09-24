import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import math

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
plt.plot(x[idx], y2[idx], 'ro')

plt.plot(x, 1/2*y1,'--', label='P(L=1|x)')
plt.plot(x, 1/2*y2,'--', label='P(L=2|x)')

plt.xlabel('x')
plt.ylabel('p(x|L=l)')

plt.title("Class-conditional PDFS and Posterior Probabilities")

plt.legend()
plt.show()