import numpy as np
from likelihood import likelihood
from prior import prior 
import time

const_e = np.exp(1)

def run_test_algorithm(n_trials, n_burnin, n_static, x0, m0, h0, priors, hyperpriors, data, n_xenolith, n_plate, n_adiabat, n_attenuation, n_viscosity):
    x = x0
    m = m0
    h = h0
    n_m = len(m)
    n_h = len(h)
    n_params = len(m) + len(h)
    prior_m = prior(priors[0,:], priors[1,:], m)
    prior_h = prior(hyperpriors[0,:], hyperpriors[1,:], h)
    prior_x = prior_m + prior_h
    likelihood_x = likelihood(data, m, h, n_xenolith, n_plate, n_adiabat, n_attenuation, n_viscosity)
    print(prior_x, likelihood_x)
    proposal_priors = (priors[1,:] / 50)**2
    proposal_hyperpriors = np.full(n_h, 0.1**2)
    proposal_covariances = np.diag(np.concatenate((proposal_priors, proposal_hyperpriors)))
    S0=np.linalg.cholesky(proposal_covariances)
    alpha_ideal = 0.234
    gamma = (2.38**2)/n_params
    model = np.zeros((n_params, n_trials))
    avg_model = np.zeros(n_params)
    track_posterior = np.zeros(n_trials)
    accepted_model = []
    n_accepted = 0

    for i in range(n_static):
        model[:,i] = x
        track_posterior[i] = prior_x + likelihood_x
        U=np.random.multivariate_normal(np.zeros(n_params), np.eye(n_params))
        y = x + np.matmul(S0,U)
        prior_m = prior(priors[0,:], priors[1,:], y[0:n_m])
        prior_h = prior(hyperpriors[0,:], hyperpriors[1,:], y[n_m:])
        prior_y = prior_m + prior_h
        likelihood_y = likelihood(data, y[0:n_m], y[n_m:], n_xenolith, n_plate, n_adiabat, n_attenuation, n_viscosity) 
        alpha = min(0, (likelihood_y - likelihood_x) + (prior_y - prior_x))
        u = np.log(np.random.uniform(low = 0, high = 1, size = 1))
        if u < alpha: 
            x = y
            prior_x = prior_y
            likelihood_x = likelihood_y
            accepted_model.append(x)
            n_accepted += 1

    avg_model = np.mean(model[:,0:n_static], axis = 1) # mean model for first (n_static - 1) trials
    C = np.cov(model[:,0:n_static]) # empirical covariance for first (n_static - 1) trials

    for i in range(n_static, n_trials): 
        print(np.abs((n_accepted / i) - alpha_ideal))
        track_posterior[i] = prior_x + likelihood_x
        model[:,i] = x
        delta = x - avg_model
        C = ((i - 2)/(i - 1)*C) + (1/i)*np.outer(delta, delta)
        avg_model = ((i - 1)*avg_model + x)/i
        U=np.random.multivariate_normal(np.zeros(n_params), np.eye(n_params))
        if gamma + ((i)**(-0.5))*(np.exp(alpha) - alpha_ideal) > 0:
            gamma += ((i)**(-0.5))*(np.exp(alpha) - alpha_ideal)
        G = gamma*C + (1e-30*np.eye(n_params))
        S = np.linalg.cholesky(G)
        y = x + np.matmul(S,U)
        prior_m = prior(priors[0,:], priors[1,:], y[0:n_m])
        prior_h = prior(hyperpriors[0,:], hyperpriors[1,:], y[n_m:])
        prior_y = prior_m + prior_h
        likelihood_y = likelihood(data, y[0:n_m], y[n_m:], n_xenolith, n_plate, n_adiabat, n_attenuation, n_viscosity) 
        alpha = min(0, (likelihood_y - likelihood_x) + (prior_y - prior_x))
        u = np.log(np.random.uniform(low = 0, high = 1, size = 1))
        if u < alpha: 
            x = y
            prior_x = prior_y
            likelihood_x = likelihood_y
            accepted_model.append(x)
            n_accepted += 1
            
    save_model = model[:, n_burnin:n_trials]
    return save_model, track_posterior