import matplotlib.pyplot as plt
import numpy as np
import sklearn as sk
import random as rd
import math
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from sklearn.neural_network import MLPRegressor
from scipy.stats import multivariate_normal

def MLP(d):
	f, n = np.shape(d)
	best = None
	error = math.inf
	result = []
	def kford_data(i, n, d,):
		start = int(i*n/10)
		end = int((i+1)*n/10)
		test = d[:,start:end]
		first = d[:,0:start]
		second = d[:,end:n]
		train = np.concatenate((first, second), axis=1 )
		return train , test 
	
	neurons = range(1,11,1)
	for neuron in neurons:
		clf = MLPRegressor(hidden_layer_sizes=(1, neuron), activation="relu", learning_rate ='invscaling', solver='lbfgs', alpha=1e-5, random_state=1)
		result.append([])
		for i in range(10):	
			train, test = kford_data(i, n, d)
			train_data = np.transpose(np.array([train[0]]))
			test_data = np.transpose(np.array([test[0]]))
			train_label = np.transpose(np.array([train[1]]))
			test_label = np.transpose(np.array([test[1]]))
			print(np.shape(test_data), np.shape(test_label))
			clf.fit(train_data, train_label)
			#decision = clf.predict(test_data)
			clf.predict(test_data)
			score = clf.score(test_data, test_label)
			#error_rate = (len(test_label) - list(decision-test_label).count(0) )/len(test_label)
			if score <= error and score > 0:
				error = score
				best = clf
			result[-1].append(score)
	plt.clf()
	plt.contour(range(10),neurons,result,20)
	plt.colorbar()
	print(result)
	plt.ylabel("neurons")
	plt.xlabel("k-ford")
	plt.title("10-ford cross validation for various neurons")
	plt.savefig("%dkford.png"%n)
	return best
	
		
if __name__=="__main__":
	plotData = 1
	n = 2
	Ntrain = 1000
	Ntest = 10000 
	alpha = np.array([0.33,0.34,0.33])
	meanVectors = np.array([[-18, 0, 18],[-8, 0, 8]])
	covEvalues = np.array([[3.2**2, 0],[0, 0.6**2]]) 
	covEvectors = np.array([[[1.0, -1.0], [1.0, 1.0]], [[1.0, 0],[0, 1.0]], [[1.0, -1.0],[1.0 ,1.0]]])
	covEvectors[0] = covEvectors[0] / math.sqrt(2)
	covEvectors[2] = covEvectors[2] / math.sqrt(2)

	t = np.random.rand(1, Ntrain)
	ind1 = ( (0 <= t) & (t <= alpha[1]))
	ind2 = ((alpha[1] < t) & (t <= alpha[1]+alpha[2]) )
	ind3 = ((alpha[1]+alpha[2] <= t) & (t <= 1))
	Xtrain = np.zeros((n, Ntrain))
	randnum = np.random.standard_normal((n,len(ind1[0])))
	Xtrain[:,ind1[0]] = np.add(np.dot(np.dot(covEvectors[0], covEvalues**(1/2)), randnum[:, ind1[0]]), meanVectors[:,0].reshape((2,1)) )
	
	randnum = np.random.standard_normal((n,len(ind2[0])))
	Xtrain[:,ind2[0]] = np.add(np.dot(np.dot(covEvectors[1], covEvalues**(1/2)), randnum[:, ind2[0]]), meanVectors[:,1].reshape((2,1)))
	randnum = np.random.standard_normal((n,len(ind3[0])))
	Xtrain[:,ind3[0]] = np.add(np.dot(np.dot(covEvectors[2], covEvalues**(1/2)), randnum[:, ind3[0]]), meanVectors[:,2].reshape((2,1)))


	t = np.random.rand(1, Ntest)
	ind1 = ( (0 <= t) & (t <= alpha[1]))
	ind2 = ((alpha[1] < t) & (t <= alpha[1]+alpha[2]) )
	ind3 = ((alpha[1]+alpha[2] <= t) & (t <= 1))
	Xtest = np.zeros((n, Ntest))
	randnum = np.random.standard_normal((n,len(ind1[0])))
	Xtest[:,ind1[0]] = np.add(np.dot(np.dot(covEvectors[0], covEvalues**(1/2)), randnum[:, ind1[0]]), meanVectors[:,0].reshape((2,1)) )
	
	randnum = np.random.standard_normal((n,len(ind2[0])))
	Xtest[:,ind2[0]] = np.add(np.dot(np.dot(covEvectors[1], covEvalues**(1/2)), randnum[:, ind2[0]]), meanVectors[:,1].reshape((2,1)))
	randnum = np.random.standard_normal((n,len(ind3[0])))
	Xtest[:,ind3[0]] = np.add(np.dot(np.dot(covEvectors[2], covEvalues**(1/2)), randnum[:, ind3[0]]), meanVectors[:,2].reshape((2,1)))

	if plotData == 1:
		plt.subplot(121)
		ax1 = plt.subplot(1,2,1)
		ax1.plot(Xtrain[0,:],Xtrain[1,:],'.')
		ax1.set(title='Training Data')
		ax2 = plt.subplot(1,2,2)
		ax2.plot(Xtest[0,:],Xtest[1,:],'.')
		ax2.set(title = 'Testing Data')
		plt.savefig("distribution.png")
	
	mlp = MLP(Xtrain)
	score = mlp.score(np.transpose(np.array([Xtest[0,:]])), np.transpose(np.array([Xtest[1,:]])))
	print(score)