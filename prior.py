import numpy as np

const_pi = np.sqrt(2*np.pi)

def prior(prior_means,prior_sigmas,y):
    prior_y = np.zeros(len(y))
    for i in range(len(y)):
        mu=prior_means[i]
        sig=prior_sigmas[i]
        dist_val=y[i]
        #prior_y[i]=np.longdouble((np.exp(-0.5*(((dist_val-mu)/sig)**2)))/(np.sqrt(2*np.pi)*sig))
        prior_y[i] = -0.5*(((dist_val-mu)/sig)**2) - np.log(const_pi*sig)
    return np.sum(prior_y)