import numpy as np
from datetime import datetime
import os 
import os.path 
import pandas as pd

def create_output_directory(f = True):
    now = datetime.now()  #takes current time and passes it to the main program, and sets up an output directory
    now = now.strftime("%Y-%m-%d_%H:%M:%S")
    if f == True:
        os.makedirs('./output/' + now)
    return now

def save_samples(samples, parameter_labels, now):
    df = pd.DataFrame(data = samples.T, columns = parameter_labels)
    df.to_csv('./output/' + now + '/samples_postburnin.csv', index = False, sep = '\t')
    #df.to_csv('./output/test/samples_postburnin.csv', index = False, sep = '\t')

#save_samples(np.zeros((3, 10)).T, ['mu0', 'dmudT', 'dmudP'], 'now')