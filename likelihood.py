import numpy as np
from thermodynamic_conversions import *

def likelihood_xenolith(data, m, h):
    # Load in file of structure Vs, sig_Vs, T, depth w/ header 
    #data = np.loadtxt(data, skiprows = 1).T 
    print(m)
    print(h)
    #Vs = data[0]
    #sig_Vs = data[1]
    #weighted_sig_Vs = (10**h)*sig_Vs
    #T = data[2]
    #depth = data[3]
    Vs = data[:,2]
    sig_Vs = 0.01 * Vs
    weighted_sig_Vs = (10**h)*sig_Vs
    T = data[:,1]
    depth = data[:,0]
    print(Vs)
    n = len(T)
    Vs_fromT = np.zeros(n)
    Vs_diff = np.zeros(n)
    RMS = 0 
    for i in range(n):
        Vs_fromT[i] = Vs_calc(m, T[i], depth[i])
        Vs_diff[i] = ((Vs[i] - Vs_fromT[i]) / weighted_sig_Vs[i])**2
        RMS += (Vs[i] - Vs_fromT[i])**2
    chi_squared = np.sum(Vs_diff)
    P = -0.5 * chi_squared - np.log(((2*np.pi)**(n/2))*np.prod(weighted_sig_Vs, dtype=np.longdouble)*1e+150) + np.log(1e+150)
    RMS = np.sqrt(RMS / n) / np.mean(sig_Vs)
    return P, RMS

def likelihood_Vs_plate(data, m, h):
    # Load in file of structure Vs, sig_Vs, T, depth w/ header 
    Vs = data[0]
    sig_Vs = data[1]
    weighted_sig_Vs = (10**h)*sig_Vs
    T = data[2]
    depth = data[3]
    n = len(T)
    Vs_fromT = np.zeros(n)
    Vs_diff = np.zeros(n)
    RMS = 0 
    for i in range(n):
        Vs_fromT[i] = Vs_calc(m, T[i], depth[i])
        Vs_diff[i] = ((Vs[i] - Vs_fromT[i]) / weighted_sig_Vs[i])**2
        RMS += (Vs[i] - Vs_fromT[i])**2
    chi_squared = np.sum(Vs_diff)
    P = -0.5 * chi_squared - np.log(((2*np.pi)**(n/2))*np.prod(weighted_sig_Vs, dtype=np.longdouble)*1e+150) + np.log(1e+150)
    RMS = np.sqrt(RMS / n) / np.mean(sig_Vs)
    return P, RMS

def likelihood_Vs_adiabat(data, m, h):
    # Load in file of structure Vs, sig_Vs, T, depth w/ header 
    Vs = data[0]
    sig_Vs = data[1]
    weighted_sig_Vs = (10**h)*sig_Vs
    T = data[2]
    depth = data[3]
    n = len(T)
    Vs_fromT = np.zeros(n)
    Vs_diff = np.zeros(n)
    RMS = 0 
    for i in range(n):
        Vs_fromT[i] = Vs_calc(m, T[i], depth[i])
        Vs_diff[i] = ((Vs[i] - Vs_fromT[i]) / weighted_sig_Vs[i])**2
        RMS += (Vs[i] - Vs_fromT[i])**2
    chi_squared = np.sum(Vs_diff)
    P = -0.5 * chi_squared - np.log(((2*np.pi)**(n/2))*np.prod(weighted_sig_Vs, dtype=np.longdouble)*1e+150) + np.log(1e+150)
    RMS = np.sqrt(RMS / n) / np.mean(sig_Vs)
    return P, RMS

def likelihood_attenuation(data, m, h):
    # Load in file of structure Q, sig_Q, Vs, depth w/ header 
    Q = data[0]
    sig_Q = data[1]
    weighted_sig_Q = (10**h)*sig_Q
    Vs = data[2]
    depth = data[3]
    n = len(Vs)
    Q_fromVs = np.zeros(n)
    Q_diff = np.zeros(n)
    RMS = 0
    for i in range(n):
        Q_fromVs[i] = Q_calc(m, Vs[i], depth[i])
        Q_diff[i] = ((Q[i] - Q_fromVs[i]) / weighted_sig_Q[i])**2
        RMS += (Q[i] - Q_fromVs[i])**2
    chi_squared = np.sum(Q_diff)
    P = -0.5 * chi_squared - np.log(((2*np.pi)**(n/2))*np.prod(weighted_sig_Q, dtype = np.longdouble))
    RMS = np.sqrt(RMS / n) / np.mean(sig_Q)
    return P, RMS

def likelihood_viscosity(data, m, h):
    # Load in file of structure eta, sig_eta, T, depth w/ header
    eta = data[0]
    sig_eta = data[1]
    weighted_sig_eta = (10**h)*sig_eta
    Vs = data[2]
    depth = data[3]
    n = len(Vs)
    eta_fromVs = np.zeros(n)
    for i in range(n):
        eta_fromVs[i] = np.log10(visc_calc(m, Vs[i], depth[i]))
    mean_eta = np.mean(eta)
    mean_eta_fromVs = np.mean(eta_fromVs)
    chi_squared = ((mean_eta - mean_eta_fromVs) / weighted_sig_eta[0])**2
    P = -0.5 * chi_squared - np.log(((2*np.pi)**(1/2))*weighted_sig_eta[0])
    RMS = np.sqrt((mean_eta - mean_eta_fromVs)**2) / sig_eta[0]
    return P, RMS


def likelihood(data, m, h, n_xenolith, n_plate, n_adiabat, n_attenuation, n_viscosity):
    P_xenolith = 0
    P_plate = 0
    P_adiabat = 0
    P_attenuation = 0
    P_viscosity = 0
    if n_xenolith > 0: 
        RMS_xenolith = 0
        #for i in range(1):
        for i in range(n_xenolith):
            P_xenolith_individual, RMS_xenolith_individual = likelihood_xenolith(data[0][i], m, h[i])
            P_xenolith += P_xenolith_individual
            RMS_xenolith += RMS_xenolith_individual
    if n_plate > 0:
        P_plate, RMS_plate = likelihood_Vs_plate(data[1], m, h[n_xenolith])
    if n_adiabat > 0:   
        P_adiabat, RMS_adiabat = likelihood_Vs_adiabat(data[2], m, h[n_xenolith + n_plate])
    if n_attenuation > 0:
        P_attenuation, RMS_attenuation = likelihood_attenuation(data[3], m, h[n_xenolith + n_plate + n_adiabat])
    if n_viscosity > 0:
        P_viscosity, RMS_viscosity = likelihood_viscosity(data[4], m, h[n_xenolith + n_plate + n_adiabat + n_attenuation])
    return P_xenolith + P_plate + P_adiabat + P_attenuation + P_viscosity