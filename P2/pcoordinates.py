import pandas as pd
from scipy.io import arff
import matplotlib.pyplot as plt
import sys

# data, meta = arff.loadarff(sys.argv[1])

# df = pd.DataFrame(data)
df = pd.read_csv(sys.argv[1])

pd.plotting.parallel_coordinates(df, 'class', color=('#556270', '#4ECDC4'))

plt.show()


