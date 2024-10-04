from ca import CASim
from pyics import paramsweep
import numpy as np
import time
import matplotlib.pyplot as plt
import csv

b4 = time.time()

sim = CASim()
paramsweep(sim,
           10,
           {'steps': 500,
            'random_forest': False,
            'wind_speed': 12,
            'density': 0.9,
            'vegetation_density': 0.7,
            'aerial_firefighting_aircrafts': 0,
            'aerial_firefighting_aoe': 0,
            'humidity': 0.43,
            'firebreak_offs': [(200, 40), (150, 50), (100, 60), (50, 70), (0, 80)],
            'firebreak_len': np.arange(0, 60, 10),
            },
           ['burned'],
           csv_base_filename='burned_fb_experiment_data',
           measure_interval=0
           )

print('experiment took: ', time.time() - b4)
