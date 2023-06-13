import numpy as np
import matplotlib.pyplot as plt
import scipy.stats

s = 60
sigma_0 = 200
sigma_1 = 10

p_0 = scipy.stats.norm(s, sigma_0^2)
p_1 = scipy.stats.norm(s, sigma_1^2)

x_all = []
y_0_all = []
y_1_all = []
for i in range(0, 120):
    y_0 = p_0.pdf(i)
    y_1 = p_1.pdf(i)

    x_all.append(i)
    y_0_all.append(y_0)
    y_1_all.append(y_1)

plt.plot(x_all, y_0_all, color='gray',linewidth=3, linestyle='dashed', label="$p(T|c=0)$")
plt.plot(x_all, y_1_all, color='k', label="$p(T|c=1)$")
plt.plot([s] * 100, np.linspace(0, 0.051, 100), color="k", linestyle='dotted')
plt.legend()
plt.xlabel("T")
plt.ylabel("$p(T|c)$")
plt.savefig("pic/pTc.pdf", format='pdf', bbox_inches='tight')
plt.show()

