import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import Lasso, Ridge, LinearRegression
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sys import stdout, argv, exit
import copy as cp
#from imageio import imread



class HomeMadeOLS():
	# Ordinaty least squared regression. Fit fit's the data
	# and predict predicts
	# ConfIntBeta finds the confidence intervar of the beta values
	# and plotter plots them

	def __init__(self):

		self.beta = 0
		self.pred = 0

	def fit(self, X, zr):
		self.beta = np.linalg.inv( X.T @ X ) @ X.T @ zr

		return self

	def predict(self, X):
		self.pred = X @ self.beta

		return self

	def ConfIntBeta(self, X, zr, pred):
		N = X.shape[0]
		p = X.shape[1]


		variance = 1./(N - p - 1) * np.sum((zr - pred)**2)

		self.var_beta = [(np.linalg.inv(X.T @ X))[i, i] for i in range(0, p)]

		self.conf_intervals = [[float(self.beta[i]) - 2*np.sqrt(self.var_beta[i]), 
							float(self.beta[i]) + 2*np.sqrt(self.var_beta[i])] for i in range(0, len(self.var_beta))]

		return self

	def plotter(self, fs1 = 20, fs2 = 20, fs3 = 20, task = 'a', method = 'OLS', patch = 'None'):

		twostd_beta = 2*np.sqrt(self.var_beta)

		plt.errorbar(np.linspace(0, len(self.beta), len(self.beta)), self.beta, yerr=twostd_beta)
		plt.title(r'$\beta$ confidence for %s' %method, fontsize = fs1)
		plt.xlabel(r'$\beta_j$', fontsize = fs2)
		plt.savefig('../figures/%s-%s-contour-beta-patch%s.png' %(task, method, patch))
		plt.tight_layout()


class HomeMadeRidge():
	# Ridge regression. Fit fit's the data
	# and predict predicts
	# ConfIntBeta finds the confidence intervar of the beta values
	# and plotter plots them

	def __init__(self):

		self.beta = 0
		self.pred = 0

	def fit(self, X, zr, lmb = 1e-3):
		self.Id_mat = np.eye(X.shape[1])
		self.beta = (np.linalg.inv(X.T @ X + lmb*self.Id_mat) @ X.T @ zr).flatten()

		return self

	def predict(self, X):
		self.pred = X @ self.beta

		return self

	def ConfIntBeta(self, X, zr, pred, lmb = 1e-3):
		N = X.shape[0]
		p = X.shape[1]

		variance = 1./(N - p - 1) * np.sum((zr - pred)**2)

		self.var_beta = [(variance*(np.linalg.inv(X.T @ X * lmb * self.Id_mat)) @ X.T @ X @((np.linalg.inv(X.T @ X * lmb * self.Id_mat)).T))[i, i] for i in range(0, p)]

		self.conf_intervals = [[float(self.beta[i]) - 2*np.sqrt(self.var_beta[i]), 
							float(self.beta[i]) + 2*np.sqrt(self.var_beta[i])] for i in range(0, len(self.var_beta))]

		return self

	def plotter(self, fs1 = 20, fs2 = 20, fs3 = 20, task = 'a', method = 'OLS', patch = 'None'):

		twostd_beta = 2*np.sqrt(self.var_beta)

		plt.errorbar(np.linspace(0, len(self.beta), len(self.beta)), self.beta, yerr=twostd_beta)
		plt.title(r'$\beta$ confidence for %s' %method, fontsize = fs1)
		plt.xlabel(r'$\beta_j$', fontsize = fs2)
		plt.savefig('../figures/%s-%s-contour-beta-patch%s.png' %(task, method, patch))
		plt.tight_layout()

class HomeMadeLogReg:

	def __init__(self, itr = 1e2, n_epochs = 5, M = 500,  eta = 1, optimizer = "SGD"):
		self.itr = itr
		self.n_epochs = n_epochs
		self.M = M
		self.eta = eta
		self.optimizer = optimizer

	def set_regularization_method(self, penalty):
		"""Set the penalty/regularization method to use."""

		self.penalty = penalty

		if penalty == "l1":
			self._get_penalty = _l1
			self._get_penalty_derivative = _l1_derivative
		elif penalty == "l2":
			self._get_penalty = _l2
			self._get_penalty_derivative = _l2_derivative
		elif isinstance(type(penalty), None):
			self._get_penalty = lambda x: 0.0
			self._get_penalty_derivative = lambda x: 0.0
		else:
			raise KeyError(("{} not recognized as a regularization"
					" method.".format(penalty)))

	def sigmoid(self, Xtheta):
		return 1./(1 + np.exp(-Xtheta))

	def log_likelihood(self, X, y, theta):
		
		ll = 0

		for i in range(0, X.shape[0]):
			ll += (y[i] * theta.T @ X[i] - np.log(1 + np.exp( theta.T @ X[i])))

		return ll

	def predict(self, X, theta):
		return X @ theta

	def fit(self, X_train, y_train):
		
		X = X_train
		y = y_train

		self.N_features, self.p = X.shape
		
		if len(y.shape) > 1:
			_, self.N_labels = y.shape

		# Adds constant term and increments the number of predictors
		X = np.hstack([np.ones((self.N_features, self.p)), X])
		self.p += 1

		# # Adds beta_0 coefficients
		# self.coef = np.zeros((self.p, self.N_labels))

		# self.coef[0, :] = np.ones(self.N_labels)

		self.coef = np.random.randn(self.p*2 - 2,1)

		self.coef = self.Optimizer(self.optimizer, X, y, self.coef, self.eta)

		for i in range(len(self.cost_value)):
			if i % 1000 == 0:
				print (self.cost_value[i])

		self.fit_performed = True


	def Optimizer(self, optimizer, X, y, theta, eta):
		self.cost_value = []

		if optimizer == "SGD":
			m = self.N_features/self.M

			for epoch in range(self.n_epochs):
				print (epoch)
				for i in range(int(m)):
					random_index = np.random.randint(m)
					Xi = X[random_index:random_index + 5]
					yi = y[random_index:random_index + 5]

					gradients = Xi.T @ (self.sigmoid(self.predict(Xi, theta)) - yi)/X.shape[0]
					#gradients = self.cost_function_gradient(Xi, yi, theta)/X.shape[0]
					#print (gradients)
					eta = self.learning_schedule(epoch * m + i)
					theta = theta - eta*gradients
					#print (theta)

					self.cost_value.append(self.cost_function(X, y, theta))
			return theta

		# elif optimizer == "GD":
		# 	for i in range(self.itr):
		# 		gradient =  / X.shape[0]
		# 		theta -= gradient*eta
		# 	return theta

		else:
			print ("Optimizer has to be SGD for now.")
			sys.exit()


	def cost_function(self, X, y, theta):
		y_pred = self.predict(X, theta)

		p_probabilities = self.sigmoid(y_pred)

		return -1/self.N_features*np.sum(y @ np.log(p_probabilities) + (1 - y)@np.log(p_probabilities))

	def learning_schedule(self, t, t0 = 5, t1 = 50):
		return t0/(t + t1)

	def predict_proba(self, X):
		"""Predicts probability of a design matrix X."""

		if not self.fit_performed:
		    raise UserWarning("Fit not performed.")

		X = np.hstack([np.ones(X.shape), X])
		probabilities = self.sigmoid(self.predict(X, self.coef))
		results = np.asarray([1 - probabilities, probabilities])
		return np.moveaxis(results, 0, 1)






class MapDataImport():
	#imports and maps data
	def __init__(self):
		b = 0

	def ImportData(self, filename = '../data/SRTM_data_Norway_1.tif'):
		self.terrain = imread(filename)

	def PlotTerrain(self):
		plt.figure()
		plt.title('Terrain over Norway 1')
		plt.imshow(self.terrain, cmap='gray')
		plt.xlabel('X')
		plt.ylabel('Y')
		plt.show()


def __test_logistic_regression():
    from sklearn import datasets
    import sklearn.linear_model as sk_model
    import matplotlib.pyplot as plt

    iris = datasets.load_iris()
    X = iris["data"][:, 3:]  # petal width
    y = (iris["target"] == 2).astype(np.int)  # 1 if Iris-Virginica, else 0

    # SK-Learn logistic regression
    sk_log_reg = sk_model.LogisticRegression(
        solver="liblinear", C=1.0, penalty="l2", max_iter=10000)
    sk_log_reg.fit(cp.deepcopy(X), cp.deepcopy(y))
    X_new = np.linspace(0, 3, 1000).reshape(-1, 1)
    y_sk_proba = sk_log_reg.predict_proba(X_new)

    print("SK-learn coefs: ", sk_log_reg.intercept_, sk_log_reg.coef_)

    # Manual logistic regression
    log_reg = HomeMadeLogReg(eta=1.0, itr=100000)

    log_reg.fit(cp.deepcopy(X), cp.deepcopy(y.reshape(-1, 1)))
    y_proba = log_reg.predict_proba(X_new)

    print("Manual coefs:", log_reg.coef)

    fig = plt.figure()

    # SK-Learn logistic regression
    ax1 = fig.add_subplot(211)
    ax1.plot(X_new, y_sk_proba[:, 1], "g-", label="Iris-Virginica(SK-Learn)")
    ax1.plot(X_new, y_sk_proba[:, 0], "b--",
             label="Not Iris-Virginica(SK-Learn)")
    ax1.set_title(
        r"SK-Learn versus manual implementation of Logistic Regression")
    ax1.set_ylabel(r"Probability")
    ax1.legend()

    # Manual logistic regression
    ax2 = fig.add_subplot(212)
    ax2.plot(X_new, y_proba[:, 1], "g-", label="Iris-Virginica(Manual)")
    ax2.plot(X_new, y_proba[:, 0], "b--", label="Not Iris-Virginica(Manual)")
    ax2.set_ylabel(r"Probability")
    ax2.legend()
    plt.show()


if __name__ == '__main__':
    __test_logistic_regression()