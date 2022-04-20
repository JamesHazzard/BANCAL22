import numpy as np
from algorithm import *
from save import *
from data import *
from prior import get_starting_model
import time
import matplotlib.pyplot as plt

t_start = time.time()

now = create_output_directory() # get current time and produce an output directory using this timestamp
print("Beginning inversion at", now)

data, n_data = get_data() # retrieve inversion data
x0, m0, h0, priors, hyperpriors = get_starting_model() # generate a starting model

n_trials = 100
n_burnin = int(0.5*n_trials)
n_static = 99
samples, track_posterior = run_test_algorithm(n_trials, n_burnin, n_static, x0, m0, h0, priors, hyperpriors, data, n_data)

t_end = time.time()
print(t_end - t_start)

m_labels = np.loadtxt('./priors.txt', max_rows = 1, dtype = str) # Get param labels for saving 
h_labels = np.loadtxt('./hyperpriors.txt', max_rows = 1, dtype = str, comments = None)[1:] # Get hyperparam labels for saving
x_labels = np.concatenate((m_labels, h_labels)) # Join labels
save_samples(samples, x_labels, now) # Save model samples