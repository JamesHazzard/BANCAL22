import numpy as np
from algorithm import *

priors = np.loadtxt('./priors.txt', skiprows = 1)
hyperpriors = np.loadtxt('./hyperpriors.txt', skiprows = 1)

# Initialise anelasticity model parameters randomly
m0 = np.random.normal(loc = priors[0,:], scale = priors[1,:])
# Initialise data hyperparameters randomly
h0 = np.random.normal(loc = hyperpriors[0,:], scale = hyperpriors[1,:])

data_type = ['xenolith', 'plate', 'adiabat', 'attenuation', 'viscosity']
data_name = ['nodule_obs_all.zTVslln', 'plate.VseTz', 'adiabat.VseTz', 'attenuation.QeVsz', 'viscosity.neVsz']
data_xenolith = []
data_plate = []
data_adiabat = []
data_attenuation = []
data_viscosity = []
data_selection = np.loadtxt('./options/data_selection.txt', skiprows = 1)
if data_selection[0] == 1:
    data_xenolith = np.loadtxt('./data/' + data_type[0] + '/'+data_name[0])
    data_xenolith = np.split(data_xenolith, np.where(np.diff(data_xenolith[:, 5]))[0] + 1)
if data_selection[1] == 1:
    data_plate = [np.loadtxt('./data/' + data_type[1] + '/' + data_name[1], skiprows = 1).T]
if data_selection[2] == 1:
    data_adiabat = [np.loadtxt('./data/' + data_type[2] + '/' + data_name[2], skiprows = 1).T]
if data_selection[3] == 1:
    data_attenuation = [np.loadtxt('./data/' + data_type[3] + '/' + data_name[3], skiprows = 1).T]
if data_selection[4] == 1:
    data_viscosity = [np.loadtxt('./data/' + data_type[4] + '/' + data_name[4], skiprows = 1).T]
data_length = [len(data_xenolith), len(data_plate), len(data_adiabat), len(data_attenuation), len(data_viscosity)]
n_xenolith = len(data_xenolith)
n_plate = len(data_plate)
n_adiabat = len(data_adiabat)
n_attenuation = len(data_attenuation)
n_viscosity = len(data_viscosity)
n_data = int(np.sum(np.asarray(data_selection)*np.asarray(data_length)))
data = [data_xenolith, data_plate, data_adiabat, data_attenuation, data_viscosity]

n_trials = 1000
n_burnin = int(0.5*n_trials)
run_test_algorithm(n_trials, n_burnin, m0, h0, priors, hyperpriors, data, n_xenolith, n_plate, n_adiabat, n_attenuation, n_viscosity)
#run_algorithm(n_trials, n_burnin, x0, priors, data_selection, data_length, data_xenolith, data_plate, data_adiabat, data_attenuation, data_viscosity)