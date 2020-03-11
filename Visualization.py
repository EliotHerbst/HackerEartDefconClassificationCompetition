import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('train.csv',
                 names=['Allied_Nations', 'Diplomatic_Meetings_Set', 'Percent_Of_Forces_Mobilized', 'Hostile_Nations',
                        'Active_Threats', 'Inactive_Threats', 'Citizen_Fear_Index', 'Closest_Threat_Distance(km)',
                        'Aircraft_Carriers_Responding', 'Troops_Mobilized(thousands)', 'DEFCON_Level', 'ID'])
features = ['Allied_Nations', 'Diplomatic_Meetings_Set', 'Percent_Of_Forces_Mobilized', 'Hostile_Nations',
            'Active_Threats', 'Inactive_Threats', 'Citizen_Fear_Index', 'Closest_Threat_Distance(km)',
            'Aircraft_Carriers_Responding', 'Troops_Mobilized(thousands)']
target = ['DEFCON_Level']

x = df.loc[:, features].values
y = df.loc[:, target].values
x = StandardScaler().fit_transform(x)

plt.scatter(x[:, 0], y)
plt.show()
