import numpy as np

def get_data():

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
        data_adiabat = [np.loadtxt('./data/' + data_type[2] + '/' + data_name[2], skiprows = 0).T]
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
    n_data = [n_xenolith, n_plate, n_adiabat, n_attenuation, n_viscosity]
    data = [data_xenolith, data_plate, data_adiabat, data_attenuation, data_viscosity]

    return data, n_data