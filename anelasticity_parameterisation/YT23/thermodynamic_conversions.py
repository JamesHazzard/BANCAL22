import numpy as np
from scipy.special import erf
from scipy import optimize

# Input parameters
# Gas constant
R=8.3145
# Reference pressure
Pr=1.5e9
# Reference temperature
TKr=1473.
T0=273.
# Grain size and reference grain size
d=1.e-3
dr=1.e-3
# Reference density
rhor=3291.
# Thermal expansivity
alphaT=3.59e-5
# Bulk modulus
bmod=115.2
# Raw frequency
freq=0.01

# Brent temperature minimization bounds
AX=0.
CX=2000.
tol=1e-3

# Other parameters (density, compressibility etc.):
rho0=3300.
a0=2.832e-5
a1=0.758e-8
K0=130.e9
KT=4.8
grun=6.
p0=3330.
AY=1.0
CY=1.6

# Set key parameters
Ab=0.664
alpha=0.38
tauP=6.e-5
Teta=0.94
beta=0.
delphi=0.
gamma=5.
lambdaphi=0.

# Set dV0 vs. P parameterisation parameters
y_a = -7.334963115431564676e-23
y_b = 7.510867653681621105e-12
y_c = 1.000184023114681908e+00

# Set solidus temperature at 50 km depth
solidus_50km = np.loadtxt('./data/potential_temperature/solidus_50km_temperature.T').astype(float)
#solidus_50km = 1380

# Set tansh parameters
Ap_x0 = 0.93128
Ap_ymin = 0.010265
Ap_ymax = 0.03
Ap_k = 68.929
sigmap_x0 = 0.97673
sigmap_ymin = 4.0
sigmap_ymax = 7.0
sigmap_k = 9.4906
Aeta_x0 = 0.94986
Aeta_ymin = 1.0
Aeta_ymax = 0.2
Aeta_k = 41.587

def funcVs(T,Vs_obs,m,dep):

  # Difference between observed and calculated temperature
  funcVs=np.abs(Vs_obs-Vs_calc(m,T,dep))
  return funcVs

def funcV0(x,Pin,K0,KT):

  # Difference between observed and calculated pressure for given value of P
  funcV0=np.abs(((K0*(3./2.)*(x**(7./3.)-x**(5./3.))*(1.+(((3./4.)*(KT-4.))*(x**(2./3.)-1.))))*1e-9)-(Pin*1e-9))
  return funcV0

def calc_tansh_continuousstep(x,x0,ymin,ymax,k):

    scale = ymax-ymin
    y = scale*((1 + np.exp(-2*k*(x-x0)))**(-1)) + ymin

    return y

def Vs_calc(m,T,dep):

  mu0 = m[0]
  dmudT = m[1]
  dmudP = m[2]
  eta0 = 10**m[3]
  E = m[4]
  Va = m[5]
  dTdz = m[6]
  sol50 = solidus_50km

  Pg=(dep/30.)
  P=Pg*1.e9
  TK=T+273.

  Tsol=sol50+(dTdz*(dep-50.))
  Tn=TK/(Tsol+273.)
  # Initialise parameters for raw Vs
  Ap = calc_tansh_continuousstep(Tn,Ap_x0,Ap_ymin,Ap_ymax,Ap_k)
  sigmap = calc_tansh_continuousstep(Tn,sigmap_x0,sigmap_ymin,sigmap_ymax,sigmap_k)
  Aeta = calc_tansh_continuousstep(Tn,Aeta_x0,Aeta_ymin,Aeta_ymax,Aeta_k)

  # Work out viscosity given A
  eta=((eta0*np.exp((E/R)*(1./TK-1./TKr))*np.exp((Va/R)*(P/TK-Pr/TKr)))*Aeta)

  # Unrelaxed compliance
  Ju=1./(1.e9*(mu0+(dmudP*Pg)+(dmudT*T)))

  # Determine input parameters for complex compliance terms
  tauM=eta*Ju
  tau=(3.*dep)/4.2
  tauS=tau/(2*np.pi*tauM)

  # Determine complex compliance terms
  J1=Ju*(1.+((Ab*(tauS**alpha))/alpha)+((np.sqrt(2.*np.pi)/2.)*Ap*sigmap*(1.-erf((np.log(tauP/tauS))/(np.sqrt(2.)*sigmap)))))

  # include pressure and temperature-dependent alpha
  dV0=y_a*P**2 + y_b*P + y_c
  alphaP0=dV0*np.exp((grun+1.)*((dV0**(-1.))-1.))
  rhoP0=p0*dV0
  intalphaT=(a0*(TK-273.))+((a1/2.)*((TK**2.)-(273.**2.)))
  rho=rhoP0*(1.-(alphaP0*intalphaT))

  # Calculate Vs
  Vs=1./(np.sqrt(rho*J1)*1000.)
  return Vs

def T_calc(m,Vs,dep):

  # Calculate temperature from Vs based on optimisation
  T=optimize.brent(funcVs,brack=(AX,CX),args=(Vs,m,dep,),tol=1e-3)

  return T

def Q_calc(m,Vs,dep):

  mu0 = m[0]
  dmudT = m[1]
  dmudP = m[2]
  eta0 = 10**m[3]
  E = m[4]
  Va = m[5]
  dTdz = m[6]
  sol50 = solidus_50km

  # Initialise parameters for attenuation
  T=optimize.brent(funcVs,brack=(AX,CX),args=(Vs,m,dep,),tol=tol)
  TK=T+273.
  Pg=(dep/30.)
  P=Pg*1.e9

  Tsol=sol50+(dTdz*(dep-50.))
  Tn=TK/(Tsol+273.)

  # Initialise parameters for raw Vs
  Ap = calc_tansh_continuousstep(Tn,Ap_x0,Ap_ymin,Ap_ymax,Ap_k)
  sigmap = calc_tansh_continuousstep(Tn,sigmap_x0,sigmap_ymin,sigmap_ymax,sigmap_k)
  Aeta = calc_tansh_continuousstep(Tn,Aeta_x0,Aeta_ymin,Aeta_ymax,Aeta_k)

  # Work out viscosity given A
  eta=((eta0*np.exp((E/R)*(1./TK-1./TKr))*np.exp((Va/R)*(P/TK-Pr/TKr)))*Aeta)

  # Unrelaxed compliance
  Ju=1./(1.e9*(mu0+(dmudP*Pg)+(dmudT*T)))

  # Determine input parameters for complex compliance terms
  tauM=eta*Ju
  tau=(3.*dep)/4.2
  tauS=tau/(2*np.pi*tauM)

  # Determine complex compliance terms
  J1=Ju*(1.+((Ab*(tauS**alpha))/alpha)+((np.sqrt(2.*np.pi)/2.)*Ap*sigmap*(1.-erf((np.log(tauP/tauS))/(np.sqrt(2.)*sigmap)))))
  J2=(Ju*(np.pi/2.)*((Ab*(tauS**alpha))+(Ap*(np.exp(-1.*(((np.log(tauP/tauS))**2.)/(2.*(sigmap**2.))))))))+(Ju*tauS)

  # Calculate Q
  Q=J2/J1
  return Q

def visc_calc(m,Vs,dep):

  mu0 = m[0]
  dmudT = m[1]
  dmudP = m[2]
  eta0 = 10**m[3]
  E = m[4]
  Va = m[5]
  dTdz = m[6]
  sol50 = solidus_50km
  
  # Initialise parameters for viscosity
  T=optimize.brent(funcVs,brack=(AX,CX),args=(Vs,m,dep,),tol=tol)
  TK=T+273.
  Pg=(dep/30.)
  P=Pg*1.e9

  # Find viscosity
  Tsol=sol50+(dTdz*(dep-50.))
  Tn=TK/(Tsol+273.)

  # Initialise parameters for raw Vs
  Aeta = calc_tansh_continuousstep(Tn,0.94986,1.0,0.2,41.587)

  # Work out viscosity
  eta=((eta0*np.exp((E/R)*(1./TK-1./TKr))*np.exp((Va/R)*(P/TK-Pr/TKr)))*Aeta)
  return eta
