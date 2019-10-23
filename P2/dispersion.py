from scipy.io import arff
import pandas as pd
import sys
import matplotlib.pyplot as plt
import plotly.express as px

# data, meta = arff.loadarff(sys.argv[1])

# df = pd.DataFrame(data)

df = pd.read_csv(sys.argv[1])
cols_to_norm = list(df.columns)
cols_to_norm = cols_to_norm[:-1]

# cols_to_norm = ['age','insu', 'mass', 'pedi', 'plas', 'preg', 'pres', 'skin']
# cols_to_norm = ['\'K\'','Al', 'Ba', 'Ca', 'Fe', 'Mg', 'Na', 'RI', 'Si']
# cols_to_norm = ['CACH','CHMAX', 'CHMIN', 'MMAX', 'MMIN', 'MYCT']
df[cols_to_norm] = df[cols_to_norm].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
df[cols_to_norm] = df[cols_to_norm].apply(lambda x: (x - x.mean()) / x.std())

fig = px.scatter_matrix(df, dimensions=cols_to_norm, color='class')

fig.show()

