import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

pi = np.pi                    # pi
m = 0.35                      # Mass
k = 5                         # Stiffness co-efficient
wn = np.sqrt(k/m)             # Natural circular frequency
Tn = 2*pi/wn                  # Natural time period
n = 11                        # No. of data points
Tmax = Tn                     # Maximum time for analysis
time = np.linspace(0,Tmax,n)  # Vector for time

df = pd.read_pickle('D:/M Tech/NIT/Assignment/Dynamics/Response_Dataframe.pkl')

mux_r = pd.MultiIndex.from_product([['F1','F2','F3'],['Ana','CDM','NM_a','NM_l']])
mux_c = pd.MultiIndex.from_product([['dr0','dr2','dr4','dr6','dr8','dr10'],['dis','vel','acc']])

df.index = mux_r
df.columns = mux_c

fig = plt.figure()
axes = fig.add_axes([0,0,2,1])
axes.plot(time,df.loc[('F2','Ana'),('dr4','dis')],label='Analytical Soln')
axes.plot(time,df.loc[('F2','CDM'),('dr4','dis')],'-.',label='Central Difference Method')
axes.plot(time,df.loc[('F2','NM_a'),('dr4','dis')],'-.',label='Newmark Method (Avg Accn)')
axes.plot(time,df.loc[('F2','NM_l'),('dr4','dis')],'-.',label='Newmark Method (Lin Accn)')
axes.set_xlabel('Time (in secs)')
axes.set_ylabel('Displacement (in inches)')
axes.set_title('Displacement vs Time')
axes.set_xlim([0,Tmax])
plt.legend()
plt.grid()
plt.show()
