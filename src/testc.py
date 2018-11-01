import numpy as np
from sklearn import linear_model
from sklearn.neural_network import MLPClassifier
import pickle,os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from classes import HomeMadeLogReg




import warnings
#Comment this to turn on warnings
warnings.filterwarnings('ignore')

np.random.seed() # shuffle random seed generator

# Ising model parameters
L=40 # linear system size
J=-1.0 # Ising interaction
T=np.linspace(0.25,4.0,16) # set of temperatures
T_c=2.26 # Onsager critical temperature in the TD limit

##### prepare training and test data sets

###### define ML parameters
num_classes=2
train_to_test_ratio=0.5 # training samples

# path to data directory
path_to_data='../data/'

# load data
file_name = "Ising2DFM_reSample_L40_T=All.pkl" # this file contains 16*10000 samples taken in T=np.arange(0.25,4.0001,0.25)
data = pickle.load(open(path_to_data+file_name,'rb')) # pickle reads the file and returns the Python object (1D array, compressed bits)
data = np.unpackbits(data).reshape(-1, 1600) # Decompress array and reshape for convenience
data=data.astype('int')
data[np.where(data==0)]=-1 # map 0 state to -1 (Ising variable can take values +/-1)

file_name = "Ising2DFM_reSample_L40_T=All_labels.pkl" # this file contains 16*10000 samples taken in T=np.arange(0.25,4.0001,0.25)
labels = pickle.load(open(path_to_data+file_name,'rb')) # pickle reads the file and returns the Python object (here just a 1D array with the binary labels)

# divide data into ordered, critical and disordered
X_ordered=data[:70000,:]
Y_ordered=labels[:70000]

X_critical=data[70000:100000,:]
Y_critical=labels[70000:100000]

X_disordered=data[100000:,:]
Y_disordered=labels[100000:]

# print (X_ordered.shape)
# print (X_disordered.shape)

# parter = 2

# X_ordered = np.array([X_ordered[parter*i, :] for i in range(0, int(np.floor(X_ordered.shape[0]/float(parter))))])
# Y_ordered = np.array([Y_ordered[parter*i] for i in range(0, int(np.floor(Y_ordered.shape[0]/float(parter))))])

# X_disordered = np.array([X_disordered[parter*i, :] for i in range(0, int(np.floor(X_disordered.shape[0]/float(parter))))])
# Y_disordered = np.array([Y_disordered[parter*i] for i in range(0, int(np.floor(Y_disordered.shape[0]/float(parter))))])

# print (X_ordered.shape)
# print (X_disordered.shape)

# X_ordered = np.random.choice(X_ordered, np.floor(len(X_ordered)/2.), replace = False)
# Y_ordered = np.random.choice(Y_ordered, np.floor(len(X_ordered)/2.), replace = False)

del data,labels

# define training and test data sets
X=np.concatenate((X_ordered,X_disordered))
Y=np.concatenate((Y_ordered,Y_disordered))

print (X)

# pick random data points from ordered and disordered states 
# to create the training and test sets
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,train_size=train_to_test_ratio)

# full data set
#X=np.concatenate((X_critical,X))
#Y=np.concatenate((Y_critical,Y))

print('X_train shape:', X_train.shape)
print('Y_train shape:', Y_train.shape)
print()
print(X_train.shape[0], 'train samples')
print(X_critical.shape[0], 'critical samples')
print(X_test.shape[0], 'test samples')

##### plot a few Ising states

# set colourbar map
cmap_args=dict(cmap='plasma_r')

# plot states
fig, axarr = plt.subplots(nrows=1, ncols=3, figsize = (16, 8))

axarr[0].imshow(X_ordered[20001].reshape(L,L),**cmap_args)
axarr[0].set_title('$\\mathrm{ordered\\ phase}$',fontsize=16)
axarr[0].tick_params(labelsize=16)

axarr[1].imshow(X_critical[10001].reshape(L,L),**cmap_args)
axarr[1].set_title('$\\mathrm{critical\\ region}$',fontsize=16)
axarr[1].tick_params(labelsize=16)

im=axarr[2].imshow(X_disordered[20001].reshape(L,L),**cmap_args)   #Was 50001
axarr[2].set_title('$\\mathrm{disordered\\ phase}$',fontsize=16)
axarr[2].tick_params(labelsize=16)

#fig.subplots_adjust(right=2.0)

#plt.show()

###### apply logistic regression

# define regularisation parameter
lmbdas=np.logspace(-5,5,11)

print ("Got here")

# preallocate data
train_accuracy=np.zeros(lmbdas.shape,np.float64)
test_accuracy=np.zeros(lmbdas.shape,np.float64)
critical_accuracy=np.zeros(lmbdas.shape,np.float64)

train_accuracy_SGD=np.zeros(lmbdas.shape,np.float64)
test_accuracy_SGD=np.zeros(lmbdas.shape,np.float64)
critical_accuracy_SGD=np.zeros(lmbdas.shape,np.float64)


logreg = HomeMadeLogReg()
logreg.fit(X_train, Y_train)
y_proba = logreg.predict_proba(X_test)

fig = plt.figure()

ax2 = fig.add_subplot(212)
ax2.plot(X_test, y_proba[:, 1], "g-", label="Iris-Virginica(Manual)")
ax2.plot(X_test, y_proba[:, 0], "b--", label="Not Iris-Virginica(Manual)")
ax2.set_ylabel(r"Probability")
ax2.legend()
plt.show()





# # loop over regularisation strength
# for i,lmbda in enumerate(lmbdas):

#     # define logistic regressor
#     logreg=linear_model.LogisticRegression(C=1.0/lmbda,random_state=1,verbose=0,max_iter=1E3,tol=1E-5)

#     print ("Got here ", i)

#     # fit training data
#     logreg.fit(X_train, Y_train)

#     print ("Got past fit", i)

#     # check accuracy
#     train_accuracy[i]=logreg.score(X_train,Y_train)
#     test_accuracy[i]=logreg.score(X_test,Y_test)
#     critical_accuracy[i]=logreg.score(X_critical,Y_critical)
    
#     print('accuracy: train, test, critical')
#     print('liblin: %0.4f, %0.4f, %0.4f' %(train_accuracy[i],test_accuracy[i],critical_accuracy[i]) )

#     # define SGD-based logistic regression
#     logreg_SGD = linear_model.SGDClassifier(loss='log', penalty='l2', alpha=lmbda, max_iter=100, 
#                                            shuffle=True, random_state=1, learning_rate='optimal')

#     # fit training data
#     logreg_SGD.fit(X_train,Y_train)

#     # check accuracy
#     train_accuracy_SGD[i]=logreg_SGD.score(X_train,Y_train)
#     test_accuracy_SGD[i]=logreg_SGD.score(X_test,Y_test)
#     critical_accuracy_SGD[i]=logreg_SGD.score(X_critical,Y_critical)
    
#     print('SGD: %0.4f, %0.4f, %0.4f' %(train_accuracy_SGD[i],test_accuracy_SGD[i],critical_accuracy_SGD[i]) )

#     print('finished computing %i/11 iterations' %(i+1))

# # plot accuracy against regularisation strength
# plt.semilogx(lmbdas,train_accuracy,'*-b',label='liblinear train')
# plt.semilogx(lmbdas,test_accuracy,'*-r',label='liblinear test')
# plt.semilogx(lmbdas,critical_accuracy,'*-g',label='liblinear critical')

# plt.semilogx(lmbdas,train_accuracy_SGD,'*--b',label='SGD train')
# plt.semilogx(lmbdas,test_accuracy_SGD,'*--r',label='SGD test')
# plt.semilogx(lmbdas,critical_accuracy_SGD,'*--g',label='SGD critical')

# plt.xlabel('$\\lambda$')
# plt.ylabel('$\\mathrm{accuracy}$')

# plt.grid()
# plt.legend()


#plt.show()
