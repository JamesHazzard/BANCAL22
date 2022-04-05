import numpy as np

def calculate_prior(x):
    return np.ones(len(x))

def calculate_likelihood(x):
    return np.ones(len(x))

def run_algorithm(n_trials, n_burnin, x0, priors, data_selection, data_length, data_xenolith, data_plate, data_adiabat, data_attenuation, data_viscosity):
        alpha_ideal = 0.234
        n_params = len(x0)
        gamma = (2.38**2)/n_params
        model = np.zeros((n_params, n_trials))
        accepted_model = []
        x = x0
        prior_x = calculate_prior(x)
        likelihood_x = calculate_likelihood(x)
        proposal_covariances = np.diag(np.concatenate((priors[1], np.full(n_params-len(priors[1]), 0.1**2))))
        S0=np.linalg.cholesky(proposal_covariances)

        for i in range(99):
            model[:,i] = x
            U=np.random.multivariate_normal(np.zeros(n_params), np.eye(n_params))
            y = x + np.matmul(S0,U)
            prior_y = calculate_prior(y)
            likelihood_y = calculate_likelihood(y) 
            alpha = min(1, np.exp(likelihood_y - likelihood_x)*(prior_y / prior_x))
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
            prior_y = calculate_prior(y)
            likelihood_y = calculate_likelihood(y) 
            alpha = min(1, np.exp(likelihood_y - likelihood_x)*(prior_y / prior_x))
            u = np.random.uniform(low = 0, high = 1, size = 1)
            if u < alpha: 
                x = y
                prior_x = prior_y
                likelihood_x = likelihood_y
                accepted_model.append(x)




