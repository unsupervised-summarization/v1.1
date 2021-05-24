import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def func(x):
    min_ratio = 0.05
    if x < min_ratio:
        x = 0.5 * (min_ratio - x) + min_ratio
    return x # -1~0


def func2(x):
    min_ratio = 0.1
    if x < min_ratio:
        return min_ratio ** (19*x)
    else:
        return -(9**(-x+0.45 * np.log(19073486328125.0 ** (-min_ratio) * (19073486328125.0 ** min_ratio - 1.0) * np.exp(2.19722457733622 * min_ratio)))) + 1


sns.set_theme(style="whitegrid")

x = np.arange(0, 1.5+1e-9, 0.001)
y = [func2(i) for i in x]
ax = sns.lineplot(x, y, color='coral', label='penalty')
ax.set(xlabel='length ratio', ylabel='penalty (-reward)')

plt.show()
