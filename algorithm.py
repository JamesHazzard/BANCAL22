import numpy as np
from likelihood import likelihood
from prior import prior 


def run_test_algorithm(n_trials, n_burnin, n_static, x0, m0, h0, priors, hyperpriors, data, n_xenolith, n_plate, n_adiabat, n_attenuation, n_viscosity):
    x = x0
    m = m0
    h = h0
    n_m = len(m)
    n_h = len(h)
    n_params = len(m) + len(h)
    prior_m = prior(priors[0,:], priors[1,:], m)
    prior_h = prior(hyperpriors[0,:], hyperpriors[1,:], h)
    prior_x = prior_m * prior_h
    likelihood_x = likelihood(data, m, h, n_xenolith, n_plate, n_adiabat, n_attenuation, n_viscosity)
    print(prior_x, likelihood_x)
    proposal_priors = (priors[1,:] / 50)**2
    proposal_hyperpriors = np.full(n_h, 0.1**2)
    proposal_covariances = np.diag(np.concatenate((proposal_priors, proposal_hyperpriors)))
    S0=np.linalg.cholesky(proposal_covariances)
    alpha_ideal = 0.234
    gamma = (2.38**2)/n_params
    model = np.zeros((n_params, n_trials))
    accepted_model = []

    for i in range(n_static):
            model[:,i] = x
            U=np.random.multivariate_normal(np.zeros(n_params), np.eye(n_params))
            y = x + np.matmul(S0,U)
            prior_m = prior(priors[0,:], priors[1,:], y[0:n_m])
            prior_h = prior(hyperpriors[0,:], hyperpriors[1,:], y[n_m:])
            prior_y = prior_m * prior_h
            likelihood_y = likelihood(data, y[0:n_m], y[n_m:], n_xenolith, n_plate, n_adiabat, n_attenuation, n_viscosity) 
            #likelihood_y = 0
            alpha = min(0, (likelihood_y - likelihood_x) + np.log(prior_y / prior_x))
            #print(alpha)
            #alpha = 1
            u = np.log(np.random.uniform(low = 0, high = 1, size = 1))
            if u < alpha: 
                print(likelihood_x + np.log(prior_x), likelihood_y + np.log(prior_y))
                x = y
                prior_x = prior_y
                likelihood_x = likelihood_y
                accepted_model.append(x)

    for i in range(n_static, n_trials): 
        model[:,i] = x
        U=np.random.multivariate_normal(np.zeros(n_params), np.eye(n_params))
        if gamma + ((i)**(-0.5))*(np.exp(alpha) - alpha_ideal) > 0:
            gamma += ((i)**(-0.5))*(np.exp(alpha) - alpha_ideal)
        C = (gamma)*np.cov(accepted_model, rowvar = False) + (1e-30*np.eye(n_params))
        S = np.linalg.cholesky(C)
        y = x + np.matmul(S,U)
        prior_m = prior(priors[0,:], priors[1,:], y[0:n_m])
        prior_h = prior(hyperpriors[0,:], hyperpriors[1,:], y[n_m:])
        prior_y = prior_m * prior_h
        likelihood_y = likelihood(data, y[0:n_m], y[n_m:], n_xenolith, n_plate, n_adiabat, n_attenuation, n_viscosity) 
        #likelihood_y = 0
        alpha = min(0, (likelihood_y - likelihood_x) + np.log(prior_y / prior_x))
        #print(alpha)
        #alpha = 1
        u = np.log(np.random.uniform(low = 0, high = 1, size = 1))
        if u < alpha: 
            print(likelihood_x + np.log(prior_x))
            x = y
            prior_x = prior_y
            likelihood_x = likelihood_y
            accepted_model.append(x)


def run_algorithm(n_trials, n_burnin, x0, priors, data_selection, data_length, data_xenolith, data_plate, data_adiabat, data_attenuation, data_viscosity):
        alpha_ideal = 0.234
        n_params = len(x0)
        gamma = (2.38**2)/n_params
        model = np.zeros((n_params, n_trials))
        accepted_model = []
        x = x0
        prior_x = prior(x)
        likelihood_x = likelihood(x)
        proposal_covariances = np.diag(np.concatenate((priors[1], np.full(n_params-len(priors[1]), 0.1**2))))
        S0=np.linalg.cholesky(proposal_covariances)
        print(proposal_covariances)
        print(S0)
        print(prior_x)
        print(likelihood_x)

        for i in range(99):
            model[:,i] = x
            U=np.random.multivariate_normal(np.zeros(n_params), np.eye(n_params))
            y = x + np.matmul(S0,U)
            prior_y = prior(y)
            likelihood_y = likelihood(y) 
            #alpha = min(1, np.exp(likelihood_y - likelihood_x)*(prior_y / prior_x))
            alpha = 1
            u = np.random.uniform(low = 0, high = 1, size = 1)
            if u < alpha: 
                x = y
                prior_x = prior_y
                likelihood_x = likelihood_y
                accepted_model.append(x)
            
        for i in range(n_trials - 99):
            model[:,i] = x
            U=np.random.multivariate_normal(np.zeros(n_params), np.eye(n_params))
            if gamma + ((i)**(-0.5))*(alpha - alpha_ideal) > 0:
                gamma += ((i)**(-0.5))*(alpha - alpha_ideal)
            C = (gamma)*np.cov(accepted_model, rowvar = False) + (1e-30*np.eye(n_params))
            S = np.linalg.cholesky(C)
            y = x + np.matmul(S,U)
            prior_y = prior(y)
            likelihood_y = likelihood(y) 
            # alpha = min(1, np.exp(likelihood_y - likelihood_x)*(prior_y / prior_x))
            alpha = 1
            u = np.random.uniform(low = 0, high = 1, size = 1)
            if u < alpha: 
                x = y
                prior_x = prior_y
                likelihood_x = likelihood_y
                accepted_model.append(x)




