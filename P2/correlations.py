import seaborn as sns
import pandas as pd
from scipy.io import arff
import matplotlib.pyplot as plt
import sys

# data, meta = arff.loadarff(sys.argv[1])

# df = pd.DataFrame(data)
df = pd.read_csv(sys.argv[1])
corr = df.corr()
sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns)

plt.show()
