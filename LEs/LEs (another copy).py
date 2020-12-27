import sys
sys.path.append("..")

import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from utils.datasets import load

rows, cols = 5, 7
ld = load("SJAFFE", return_X_y=False).label_distribution
fig, axes = plt.subplots(rows, cols)
count = 0
for i in range(rows):
    for j in range(cols):
        sns.barplot(x=list(range(ld.shape[1])), y=ld[count], ax=axes[i][j])
        count += 1
plt.show()