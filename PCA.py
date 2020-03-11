import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('train.csv',
                 names=['Allied_Nations', 'Diplomatic_Meetings_Set', 'Percent_Of_Forces_Mobilized', 'Hostile_Nations',
                        'Active_Threats', 'Inactive_Threats', 'Citizen_Fear_Index', 'Closest_Threat_Distance(km)',
                        'Aircraft_Carriers_Responding', 'Troops_Mobilized(thousands)', 'DEFCON_Level', 'ID'])

with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(df.head())
features = ['Allied_Nations', 'Diplomatic_Meetings_Set', 'Percent_Of_Forces_Mobilized', 'Hostile_Nations',
            'Active_Threats', 'Inactive_Threats', 'Citizen_Fear_Index', 'Closest_Threat_Distance(km)',
            'Aircraft_Carriers_Responding', 'Troops_Mobilized(thousands)']
target = ['DEFCON_Level']

x = df.loc[:, features].values
y = df.loc[:, target].values
x = StandardScaler().fit_transform(x)

with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(pd.DataFrame(data=x, columns=features).head())

pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)

principalDf = pd.DataFrame(data=principalComponents
                           , columns=['principal_component_1', 'principal_component_2'])

with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(principalDf.head(5))
    print(df[['DEFCON_Level']].head())

finalDf = pd.concat([principalDf, df[['DEFCON_Level']]], axis=1)

with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(finalDf.head(5))

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('Principal Component 1', fontsize=15)
ax.set_ylabel('Principal Component 2', fontsize=15)
ax.set_title('2 Component PCA', fontsize=20)

targets = [1, 2, 3, 4, 5]
colors = ['r', 'g', 'b', 'm', 'y']

for target, color in zip(targets, colors):
    indicesToKeep = finalDf['DEFCON_Level'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal_component_1']
               , finalDf.loc[indicesToKeep, 'principal_component_2']
               , c=color
               , s=50)
ax.legend(targets)
ax.grid()
plt.show()

print(pca.explained_variance_ratio_)
