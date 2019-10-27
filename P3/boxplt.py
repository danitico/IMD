import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('results.csv', header=0)

df.boxplot()
plt.ylabel('Accuracy')
plt.show()

