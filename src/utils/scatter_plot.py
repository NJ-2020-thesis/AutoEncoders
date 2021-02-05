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
ax.grid(color='#d3d3d3', linestyle='--', linewidth=0.8)

N = 50
x = np.random.rand(N)
y = np.random.rand(N)
colors = np.random.rand(N)
area = (30 * np.random.rand(N))**2  # 0 to 15 point radii

plt.scatter(x, y, s=area, c='red', alpha=0.5, marker="*",label='Failure')
plt.scatter(x, y, s=area, c='green', alpha=0.5, marker=".",label='Success')

plt.legend()
plt.show()