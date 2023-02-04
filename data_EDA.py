import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_rows', None)
df = pd.read_csv("cleaned_train.csv")

# plot heatmap
plt.figure(figsize=(25, 10))
sns.heatmap(df.corr())

important_columns = list(df.corr()["SalePrice"][(df.corr()["SalePrice"] > 0.50) | (df.corr()["SalePrice"] < -0.50)].index)

# plot pairplot
sns.pairplot(df[important_columns])

# subplots with violin and box plots

RL = df.query("MSZoning=='RL'")['SalePrice']
RM = df.query("MSZoning=='RM'")['SalePrice']
C = df.query("MSZoning=='C (all)'")['SalePrice']
FV = df.query("MSZoning=='FV'")['SalePrice']
RH = df.query("MSZoning=='RH'")['SalePrice']
labels = ["RL", "RM", "C", "FV", "RH"]


fig, ax = plt.subplots(2, figsize=(25, 10))
ax[0].boxplot([RL, RM, C, FV, RH], labels=labels)
ax[1].violinplot([RL, RM, C, FV, RH])
plt.show()
