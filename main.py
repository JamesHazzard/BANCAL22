import numpy as np
import shutil

data_type = ['xenolith', 'plate', 'adiabat', 'attenuation', 'viscosity']
data_name = ['nodule_obs_all.zTVslln', '', '', '', '']
data_xenolith = np.zeros(1)
data_plate = np.zeros(1)
data_adiabat = np.zeros(1)
data_attenuation = np.zeros(1)
data_viscosity = np.zeros(1)
data_selection = np.loadtxt('./data/data_selection.txt', skiprows = 1)
if data_selection[0] == 1:
    data_xenolith = np.loadtxt('./data/'+data_type[0]+'/'+data_name[0])
    data_xenolith = np.split(data_xenolith, np.where(np.diff(data_xenolith[:,5]))[0]+1)

param_selection = str(np.genfromtxt('./anelasticity_parameterisation/parameterisation_selection.txt', dtype='str'))
priors = np.loadtxt('./anelasticity_parameterisation/'+param_selection+'/priors.txt', skiprows = 1)
shutil.copyfile('./anelasticity_parameterisation/'+param_selection+'/thermodynamic_conversions.py', './likelihood/thermodynamic_conversions.py')


