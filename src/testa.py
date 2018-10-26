import numpy as np
import scipy.sparse as sp
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from functions import bootstrap
from classes import HomeMadeOLS, HomeMadeRidge
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
np.random.seed(12)

def ising_energies(states,L):
	"""
	This function calculates the energies of the states in the nn Ising Hamiltonian
	"""
	J=np.zeros((L,L),)
	for i in range(L):
		J[i,(i+1) %L]-=1.0    #What does %L do?
	# compute energies
	E = np.einsum('...i,ij,...j->...',states,J,states)

	return E


### define Ising model aprams
# system size
L=40
n = 100
test_size = 0.4


# create n random Ising states
states=np.random.choice([-1, 1], size=(n,L))

X = np.zeros((n, L*L))

print (X.shape)

for i in range(n):
	X[i] = np.outer(states[i], states[i]).ravel()

# # # 3D-versjon av X[i], tror jeg.
# # reshape Ising states into RL samples: S_iS_j --> X_p
# states=np.einsum('...i,...j->...ij', states, states)
# shape=states.shape
# states=states.reshape((shape[0],shape[1]*shape[2]))
# # build final data set
# Data=[states,energies]


# calculate Ising energies
energies=ising_energies(states,L)

X_train, X_test, E_train, E_test = train_test_split(X, energies, test_size=test_size)


#For OLS I need to implement SVD
ridge = HomeMadeRidge()
ridge.fit(X_train, E_train)
J_ridge = ridge.beta.reshape(L, L)

ridgeSK = linear_model.Ridge()

ridgeSK.set_params(alpha=1e-3) # set regularisation parameter
ridgeSK.fit(X_train, E_train) # fit model 
coefs_ridgeSK = ridgeSK.coef_ # store weights




cmap_args=dict(vmin=-1., vmax=1., cmap='seismic')

fig, axarr = plt.subplots(nrows=1, ncols=2)

im = axarr[0].imshow(J_ridge,**cmap_args)
axarr[0].set_title('$\\mathrm{Ridge}$',fontsize=16)
axarr[0].tick_params(labelsize=16)

axarr[1].imshow(coefs_ridgeSK.reshape(L, L),**cmap_args)
axarr[1].set_title('$\\mathrm{Ridge} SK$',fontsize=16)
axarr[1].tick_params(labelsize=16)

divider = make_axes_locatable(axarr[-1])
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar=fig.colorbar(im, cax=cax)

cbar.ax.set_yticklabels(np.arange(-1.0, 1.0+0.25, 0.25),fontsize=14)
cbar.set_label('$J_{i,j}$',labelpad=-40, y=1.12,fontsize=16,rotation=0)

plt.show()