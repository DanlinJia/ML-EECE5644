import numpy as np
import math

def linearTrans(z, m, v):
	# z is an guassian distribution with mean=0, variance=I
	# m and v is the desired mean and variance
	# ret = a * z + b
	b = m
	#compute the eigenvalue and eigenvector of v
	value, vector = np.linalg.eig(v)
	#sqrt the eigenvalue
	sqrt_value = np.array([ math.sqrt(i) for i in value])
	# update eigen vector with sqrted eigenvalue 
	for i in range(len(vector)):
		vector[i] = sqrt_value[i]*vector[i]
	a = vector
	tmp = a.dot(z)
	for i in range(len(b)):
		ret[i] = ret[i] + np.array([b[i] for element in range(len(ret[i])) ])
	return ret
