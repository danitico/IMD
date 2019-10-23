from scipy.io import arff
import sys
import pandas as pd
from sklearn.decomposition import PCA
import plotly.express as px

# data, meta = arff.loadarff(sys.argv[1])

# df = pd.DataFrame(data)
df = pd.read_csv(sys.argv[1])
cols_to_norm = list(df.columns)
cols_to_norm = cols_to_norm[:-1]
# cols_to_norm = ['age','insu', 'mass', 'pedi', 'plas', 'preg', 'pres', 'skin']
# cols_to_norm = ['\'K\'','Al', 'Ba', 'Ca', 'Fe', 'Mg', 'Na', 'RI', 'Si']
# cols_to_norm = ['CACH','CHMAX', 'CHMIN', 'MMAX', 'MMIN', 'MYCT']

pca = PCA(n_components=len(cols_to_norm) - 1)
f = pca.fit_transform(df[cols_to_norm])
newdf = pd.DataFrame(f)
newdf['class'] = df['class'].copy()

fig = px.scatter_matrix(newdf, dimensions=range(0, len(cols_to_norm)-1), color='class')

fig.show()
