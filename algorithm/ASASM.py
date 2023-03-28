import numpy as np
from likelihood import likelihood
from prior import prior 
import time
from scipy.stats import norm
import math

const_e = np.exp(1)

def forget_func(n):

    return math.floor(0.3 * np.sqrt(2 * n))

def run_test_algorithm(n_trials, n_burnin, n_static, x0, m0, h0, priors, hyperpriors, data, n_data):
    n_xenolith = n_data[0]
    n_plate = n_data[1]
    n_adiabat = n_data[2]
    n_attenuation = n_data[3]
    n_viscosity = n_data[4]
    x = x0
    m = m0
    h = h0
    n_m = len(m)
    n_h = len(h)
    if n_viscosity > 0:
        n_h_RMS = n_h + 1 # if viscosity data set being used, track RMS, but do not use hyperparameter on this data set
    else: 
        n_h_RMS = n_h # if no viscosity data set, RMS array is simply same length as hyperparameter array
    n_params = len(m) + len(h)
    prior_m = prior(priors[0,:], priors[1,:], m)
    prior_h = prior(hyperpriors[0,:], hyperpriors[1,:], h)
    prior_x = prior_m + prior_h
    likelihood_x, RMS_x = likelihood(data, m, h, n_xenolith, n_plate, n_adiabat, n_attenuation, n_viscosity)
    proposal_priors = (priors[1,:] / 50)**2
    proposal_hyperpriors = np.full(n_h, 0.1**2)
    proposal_covariances = np.diag(np.concatenate((proposal_priors, proposal_hyperpriors)))
    S0=np.linalg.cholesky(proposal_covariances)
    model = np.zeros((n_params, n_trials))
    RMS = np.zeros((n_h_RMS, n_trials))
    avg_model = np.zeros(n_params)
    track_posterior = np.zeros((1, n_trials))
    accepted_model = []
    n_accepted = 0
    t_init=time.time()

    a = 0.234
    d = n_params
    c = (2.38**2) / d
    v0 = 0
    scale_start = 1
    scale = scale_start
    trial = 5/(a * (1 - a))
    A = norm.ppf(norm.cdf(a / 2))
    delta = (1 - (1 / d)) * ((np.sqrt(2 * np.pi) * np.exp((A ** 2) / 2)) / (2 * A)) + (1 / (d * a * (1 - a)))
    sigma = proposal_covariances

    i = 0
    model[:,i] = x
    RMS[:,i] = np.log10(RMS_x)
    track_posterior[0,i] = prior_x + likelihood_x

    for i in range(1, n_trials):

        if i%100 == 0: 
            print(i, "%.2f" % (time.time() - t_init), "%.3f" % x[3], "%.5f" % (np.abs((n_accepted / i) - a)), "%.1f" % (prior_x + likelihood_x), ["{0:0.2f}".format(np.log10(RMS_x[y] - x[n_m + y])) for y in range(len(x[n_m:]))])
            t_init = time.time()

        scaled_sigma = scale * c * sigma
        S0 = np.linalg.cholesky(scaled_sigma)
        U=np.random.multivariate_normal(np.zeros(n_params), np.eye(n_params))
        y = x + np.matmul(S0, U)

        prior_m = prior(priors[0,:], priors[1,:], y[0:n_m])
        prior_h = prior(hyperpriors[0,:], hyperpriors[1,:], y[n_m:])
        prior_y = prior_m + prior_h
        likelihood_y, RMS_y = likelihood(data, y[0:n_m], y[n_m:], n_xenolith, n_plate, n_adiabat, n_attenuation, n_viscosity) 
        alpha = min(0, (likelihood_y - likelihood_x) + (prior_y - prior_x))
        u = np.log(np.random.uniform(low = 0, high = 1, size = 1))
        if u < alpha: 
            x = y
            prior_x = prior_y
            likelihood_x = likelihood_y
            RMS_x = RMS_y
            accepted_model.append(x)
            n_accepted += 1

        model[:,i] = x
        RMS[:,i] = np.log10(RMS_x)
        track_posterior[0,i] = prior_x + likelihood_x

        if i == 1:

            avg_model = 0.5 * (x + x0)
            sigma = (1 / (v0 + d + 3)) * (np.outer(x0, x0) + np.outer(x, x) - (2 * np.outer(avg_model, avg_model)) + ((v0 + d + 1) * sigma))

        elif forget_func(i) == forget_func(i - 1):

            old_avg_model = avg_model 
            avg_model = (((i - forget_func(i)) / (i - forget_func(i) + 1)) * avg_model) + ((1 / (i - forget_func(i) + 1)) * x)
            sigma = (1 / (i - forget_func(i) + v0 + d + 2)) * (((i - forget_func(i) + v0 + d + 1) * sigma) + np.outer(x, x) + ((i - forget_func(i)) * np.outer(old_avg_model, old_avg_model)) - ((i - forget_func(i) + 1) * np.outer(avg_model, avg_model)))

        elif forget_func(i) == forget_func(i - 1) + 1:

            old_avg_model = avg_model
            x_forget =  model[:,int(forget_func(i) - 1)]
            avg_model = avg_model + (1 / (i - forget_func(i) + 1)) * (x - x_forget)
            sigma = sigma + (1 / (i - forget_func(i) + v0 + d + 2)) * (np.outer(x, x) - np.outer(x_forget, x_forget) + ((i - forget_func(i) + 1) * (np.outer(old_avg_model, old_avg_model) - np.outer(avg_model, avg_model))))

        scale = np.amax(np.array([1e-5, scale * np.exp((delta / (trial + i) * (np.exp(alpha) - a)))]))
        if np.abs(np.log(scale) - np.log(scale_start)) > np.log(3):
            scale_start = scale
            trial = (5 / (a * (1 - a))) - i

    return model, RMS, track_posterior