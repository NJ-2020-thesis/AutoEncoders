import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg', force=True)

import numpy as np
import seaborn as sns

# Save a palette to a variable:
palette = sns.color_palette("bright")

# create data
y = np.cumsum(np.random.randn(1000, 1))
x = [i for i in range(0,len(y))]

fig, ax = plt.subplots()

ax.set_title('axes title',fontsize=12, fontweight='bold')
ax.set_xlabel('xlabel' ,fontsize=10)
ax.set_ylabel('ylabel',fontsize=10)
ax.grid(color='#d3d3d3', linestyle='--', linewidth=0.8)

# use the plot function
sns.lineplot(x=x, y=y,
            palette =palette,
            color = 'red',
            dashes=False,
            markers=False,
            label="nj"
            )

plt.show()