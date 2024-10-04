"""
Example used: https://matplotlib.org/2.0.2/examples/pylab_examples/barchart_demo.html
Python file for generating a bar plot for burned percentage experiment.
Usage: python3 plots2.py."""

import csv
import matplotlib.pyplot as plt
import numpy as np

# Number of repetitions, necessary for data processing.
reps = 10
vals = 60

bar_width = 1.5

data = None
with open('burned_fb_experiment_data_0.csv', 'r') as f:
    reader = csv.reader(f)
    data = list(reader)

header = data.pop(0)

locss = [(200, 40), (150, 50), (100, 60), (50, 70), (0, 80)]
locs = [0, 60, 120, 180, 240]
x = np.arange(0, 60, 10)

error_config = {'ecolor': '0.3'}
colors = []

counter = -2
for k, loc in enumerate(locs):
    ys = np.array([data[k + i][-1]
                   for i in range(vals)], dtype=float).reshape(6, 10)

    mean, std = [np.mean(y) for y in ys], [np.std(y) for y in ys]

    bars = plt.bar([0 + counter*bar_width, 10 + counter*bar_width, 20 + counter*bar_width, 30 + counter*bar_width, 40 + counter*bar_width, 50 + counter*bar_width], mean, bar_width, alpha=0.6,
                   yerr=std, error_kw=error_config, label=str(locss[k]))
    counter += 1


plt.xticks([0, 10, 20, 30, 40, 50], [str(x) for x in x])
plt.title('Percentage burnt vegetation as a function of firebreak length')
plt.xlabel('Firebreak length')
plt.ylabel('Percentage burned')
plt.legend()
plt.show()
