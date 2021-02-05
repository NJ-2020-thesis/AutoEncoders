import numpy as np
import seaborn as sns

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg', force=True)

palette = sns.color_palette("bright")

fig, ax = plt.subplots()
ax.set_title('axes title',fontsize=12, fontweight='bold')
ax.set_xlabel('xlabel' ,fontsize=10)
ax.set_ylabel('ylabel',fontsize=10)

plt.minorticks_on()
ax.grid(which='major',color='#d3d3d3', linestyle='-', linewidth=0.8)
ax.grid(b=True, which='minor', color='#d3d3d3', linestyle='--', alpha=0.2)


plt.xlim(-10, 20)
plt.ylim(-10, 20)

plt.legend(loc='upper left')
plt.show()