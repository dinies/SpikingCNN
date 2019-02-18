import matplotlib

matplotlib.use("qt5agg")

import matplotlib.pyplot as plt

import numpy as np

# import matplotlib.rcsetup as rcsetup
# print(rcsetup.all_backends)
# print( matplotlib.get_backend())

# print(matplotlib.__file__)
# print(matplotlib.get_configdir())

# Data for plotting
t = np.arange(0.0, 2.0, 0.01)
s = 1 + np.sin(2 * np.pi * t)

fig, ax = plt.subplots()
ax.plot(t, s)

ax.set(xlabel='time (s)', ylabel='voltage (mV)',
       title='About as simple as it gets, folks')
ax.grid()

fig.savefig("test.png")
plt.show()
