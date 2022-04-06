import numpy as np

def prior(prior_means,prior_sigmas,y):
    prior_y=np.zeros(len(y))
    for i in range(len(y)):
        mu=prior_means[i]
        sig=prior_sigmas[i]
        dist_val=y[i]
        prior_y[i]=np.longdouble((np.exp(-0.5*(((dist_val-mu)/sig)**2)))/(np.sqrt(2*np.pi)*sig))
    return np.prod(prior_y)