import numpy as np
import shutil
from GASWAM import * 

# Initialise anelasticity model parameters randomly
m0 = np.random.normal(loc = priors[0,:], scale = priors[1,:])
# Initialise data hyperparameters randomly
h0 = np.random.normal(loc = 0.0, scale = 1.0, size = n_data)
# Initialise model
x0 = np.concatenate((m0, h0))
print(x0)

n_trials = 1000
n_burnin = int(0.5*n_trials)
run_algorithm(n_trials, n_burnin, x0, priors, data_selection, data_length, data_xenolith, data_plate, data_adiabat, data_attenuation, data_viscosity)