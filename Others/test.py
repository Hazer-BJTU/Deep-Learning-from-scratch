import numpy as np
import matplotlib.pyplot as plt

x = np.random.normal(loc=6, scale=3, size=5000)
y = np.random.normal(loc=8, scale=3, size=5000)
z = np.random.normal(loc=10, scale=5, size=5000)
w = np.array([x, y])
t = np.max(w, axis=0) + z
print(t)


plt.hist(t, bins=30, color='skyblue', alpha=0.8)
plt.title('Monte Carlo analysis')
plt.xlabel('Total duration')
plt.ylabel('Frequency')
plt.show()
