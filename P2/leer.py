from scipy.io import arff
import sys
import pandas as pd
import matplotlib.pyplot as plt


# data, meta = arff.loadarff(sys.argv[1])

#df = pd.DataFrame(data)

df = pd.read_csv(sys.argv[1])
cols_to_norm = list(df.columns)
cols_to_norm = cols_to_norm[:-1]

# cols_to_norm = ['age','insu', 'mass', 'pedi', 'plas', 'preg', 'pres', 'skin']
# cols_to_norm = ['\'K\'','Al', 'Ba', 'Ca', 'Fe', 'Mg', 'Na', 'RI', 'Si']
# cols_to_norm = ['CACH','CHMAX', 'CHMIN', 'MMAX', 'MMIN', 'MYCT']
df[cols_to_norm] = df[cols_to_norm].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
df[cols_to_norm] = df[cols_to_norm].apply(lambda x: (x - x.mean()) / x.std())


df.hist()
plt.show()

