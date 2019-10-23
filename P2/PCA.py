from scipy.io import arff
import sys
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# data, meta = arff.loadarff(sys.argv[1])

# df = pd.DataFrame(data)
df = pd.read_csv(sys.argv[1])
cols_to_norm = list(df.columns)
cols_to_norm = cols_to_norm[:-1]
# cols_to_norm = ['age','insu', 'mass', 'pedi', 'plas', 'preg', 'pres', 'skin']
# cols_to_norm = ['\'K\'','Al', 'Ba', 'Ca', 'Fe', 'Mg', 'Na', 'RI', 'Si']
# cols_to_norm = ['CACH','CHMAX', 'CHMIN', 'MMAX', 'MMIN', 'MYCT']
# df[cols_to_norm] = df[cols_to_norm].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
# df[cols_to_norm] = df[cols_to_norm].apply(lambda x: (x - x.mean()) / x.std())

pca = PCA(n_components=len(cols_to_norm) - 1)

f = pca.fit_transform(df[cols_to_norm])

newdf = pd.DataFrame(f)
oldcorr = df[cols_to_norm].corr().round(3).to_numpy()

print("~ ", end='')
for i in cols_to_norm:
    print("& ", i, " ", end='')
print("\\cr")

for i in range(0, oldcorr.shape[0]):
    print(cols_to_norm[i], " ", end='')
    for j in range(0, oldcorr.shape[1]):
        print("& ", oldcorr[i,j], " ", end='')
    print("\\cr")


newcorr = newdf.corr().round(3).to_numpy()
print("~ ", end='')
for i in range(0, len(cols_to_norm)):
    print("& ", i, " ", end='')
print("\\cr")

for i in range(0, newcorr.shape[0]):
    print(i, " ", end='')
    for j in range(0, newcorr.shape[1]):
        print("& ", newcorr[i,j], " ", end='')
    print("\\cr")


newdf.hist()
plt.show()


