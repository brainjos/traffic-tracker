import numpy as np
import motmetrics as mm

print(mm.metrics.motchallenge_metrics)

data = np.load('total_metrics.npy')
acc = data[:, 13]
best = np.argsort(acc)
print(best[-5:])