import matplotlib.pyplot as plt
import numpy as np
import sklearn as sk
import random as rd
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from sklearn.neural_network import MLPClassifier
from scipy.stats import multivariate_normal

def geneData(n, m , sigma, classPriors, plot=False):
	distribution = np.zeros(4)
	for i in range(n):
		r = rd.random()
		if r<=0.1:
			distribution[0]+=1
		elif r>0.1 and r<=0.5:
			distribution[1]+=1
		elif r>0.5 and r<=0.7:
			distribution[2]+=1
		else:
			distribution[3]+=1
	
	x1=np.random.multivariate_normal(m[0], sigma[0], (int(distribution[0])))
	y1=np.ones(len(x1))
	x2=np.random.multivariate_normal(m[1], sigma[1], (int(distribution[1])))
	y2=2*np.ones(len(x2))
	x3=np.random.multivariate_normal(m[2], sigma[2], (int(distribution[2])))
	y3=3*np.ones(len(x3))
	x4=np.random.multivariate_normal(m[3], sigma[3], (int(distribution[3])))
	y4=4*np.ones(len(x4))
	x = np.concatenate((x1,x2,x3,x4))
	y = np.concatenate((y1,y2,y3,y4))
	if (plot):
		ax = plt.axes(projection='3d')
		ax.scatter3D(x1[:,0],x1[:,1],x1[:,2], cmap='Greens');
		ax.scatter3D(x2[:,0],x2[:,1],x2[:,2], cmap='Reds');
		ax.scatter3D(x3[:,0],x3[:,1],x3[:,2], cmap='Yellows');
		ax.scatter3D(x4[:,0],x4[:,1],x4[:,2], cmap='Blues');
		ax.set_xlabel('x1 ')
		ax.set_ylabel('x2 ')
		ax.set_zlabel('x3 ')
		ax.legend(('Class 1','Class 2', 'Class3', 'Class4'))
		ax.set_title("Data and their true labels")
		plt.savefig("datadistribute.png")
	return x,y
	
	
def plotEstimation(data, decision, label, title):
	ind1 = ((decision==1) & (label==1))
	ind2 = ((decision==2) & (label==2))
	ind3 = ((decision==3) & (label==3))
	ind4 = ((decision==4) & (label==4))
	ind1w = ((decision==1) & (label!=1))
	ind2w = ((decision==2) & (label!=2))
	ind3w = ((decision==3) & (label!=3))
	ind4w = ((decision==4) & (label!=4))
	plt.cla()
	ax1 = plt.axes(projection='3d')
	ax1.scatter3D(data[ind1,0],data[ind1,1],data[ind1,2], c='g', marker=".");
	ax1.scatter3D(data[ind2,0],data[ind2,1],data[ind2,2], c='g', marker=".");
	ax1.scatter3D(data[ind3,0],data[ind3,1],data[ind3,2], c='g', marker=".");
	ax1.scatter3D(data[ind4,0],data[ind4,1],data[ind4,2], c='g', marker=".");
	
	ax1.scatter3D(data[ind1w,0],data[ind1w,1],data[ind1w,2], c='k', marker="8");
	ax1.scatter3D(data[ind2w,0],data[ind2w,1],data[ind2w,2], c='r', marker="s");
	ax1.scatter3D(data[ind3w,0],data[ind3w,1],data[ind3w,2], c='y', marker="P");
	ax1.scatter3D(data[ind4w,0],data[ind4w,1],data[ind4w,2], c='b', marker="p");
	ax1.set_xlabel('x1 ')
	ax1.set_ylabel('x2 ')
	ax1.set_zlabel('x3 ')
	ax1.legend(('Correct decisions for data from Class 1', 'Correct decisions for data from Class 2',  'Correct decisions for data from Class 3', 'Correct decisions for data from Class 4', 'Wrong decisions for data from Class 1','Wrong decisions for data from Class 2','Wrong decisions for data from Class 3','Wrong decisions for data from Class 4'))
	ax1.set_title(title)
	plt.savefig(title+".png",figsize=(16, 16),dpi=160)
	return list(ind1w+ind2w+ind3w+ind4w).count(True)

def MAPClassifier(data, label, mu, sigma):
	n, features = data.shape
	decision = np.zeros(n)
	for i in range(n):
		p = 0
		for j in range(len(mu)):
			if p < multivariate_normal.pdf(data[i],mu[j],sigma[j]):
				p = multivariate_normal.pdf(data[i],mu[j],sigma[j])
				decision[i] = j+1
	e = plotEstimation(data, decision, label, "MAP: Data and their classifier decisions versus true labels")
	return decision, e
	

def MLP(data, label):
	n = len(label)
	label=np.array([list(label)])
	d = np.concatenate((data, label.T), axis=1)
	np.random.shuffle(d)
	best = None
	error = 1
	result = []
	def kford_data(i, n, d,):
		start = int(i*n/10)
		end = int((i+1)*n/10)
		test = d[start:end]
		train = np.array(list(d[0:start])+list(d[end:n]))
		return train , test 
	
	neurons = range(1,11,1)
	for neuron in neurons:
		clf = MLPClassifier(hidden_layer_sizes=(neuron, 1), activation="logistic", learning_rate ='invscaling', solver='lbfgs', alpha=1, random_state=1)
		result.append([])
		for i in range(10):	
			train, test = kford_data(i, n, d)
			train_data = train[:,0:3]
			test_data = test[:,0:3]
			train_label = train[:,3]
			test_label = test[:,3]
			clf.fit(train_data, train_label)
			decision = clf.predict(test_data)
			error_rate = (len(test_label) - list(decision-test_label).count(0) )/len(test_label)
			if error_rate <= error:
				error = error_rate
				best = clf
			result[-1].append(1-error)
	plt.clf()
	plt.contour(range(10),neurons,result,20)
	plt.colorbar()
	print(result)
	plt.ylabel("neurons")
	plt.xlabel("k-ford")
	plt.title("%d: 10-ford cross validation for various neurons"%n)
	plt.savefig("%dkford.png"%n)
	return best
	
		
if __name__=="__main__":
	N=10000
	scale = 1			# parameter to scale variance
	classPriors = [0.1,0.4,0.2, 0.3]
	m=[(-1,0,1),(-1,-2,1),(2,1,1),(1.5,-1,2)]
	sigma=[]
	sigma1=scale*np.array([[1.9, -0.4, -0.5],[-0.4, 0.2, 0.3],[-0.5, 0.3, 1.1]])
	sigma2=scale*np.array([[1.3644, 1.0038, 1.2629],[1.0038, 1.2225, 1.3130,],[ 1.2629, 1.3130, 1.5345]])
	sigma3=scale*np.array([[0.5966, 0.7130, 0.7354],[0.7130, 0.9637, 0.9712], [0.7354 ,0.9712, 1.3236]])
	sigma4=scale*np.array([[1.6509, 0.7610, 0.4427],[ 0.7610, 0.7155, 0.2141],[ 0.4427 ,0.2141, 0.5502]])
	sigma=[sigma1,sigma2,sigma3,sigma4]
	test_data, test_label = geneData(N, m , sigma, classPriors, True)
	mapdecision, maperrors = MAPClassifier(test_data, test_label, m, sigma)
	print("error for MAP is %d"%maperrors)
	for datasize in [100, 1000, 10000]:
		data, label = geneData(datasize, m , sigma, classPriors)
		mlp = MLP(data, label)
		decision = mlp.predict(test_data)
		e = plotEstimation(test_data, decision, test_label, "MLP(%d): Data and their classifier decisions versus true labels"%datasize)
		print("error for %d is %d"%(datasize, e))
		
#	mlpdecision, mlperrors = MLP(data, label, m, sigma, classPriors)