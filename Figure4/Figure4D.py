import numpy as np
import matplotlib.pyplot as plt

def generate_portion(s, mu_c, sigma_s, sigma_c):
    a = 1 / 2 / sigma_s ** 2 + 1 / 2 / sigma_c ** 2

    portion = np.exp(- (s-mu_c)**2 / (2 * sigma_c ** 2 + 2 * sigma_s ** 2)) / np.sqrt(a) / sigma_s ** 2 / sigma_c ** 2
    return portion

sigma_e = 4
sigma_c_0 = 200
sigma_c_1 = 10 # 10 # 200
mu_c_0 = 150 # 60
mu_c_1 = 50

x_all = []
y_0_all = []
y_1_all = []

for e in range(-10, 110):
    p_c_0_e = generate_portion(e, mu_c_0, sigma_e, sigma_c_0)
    p_c_1_e = generate_portion(e, mu_c_1, sigma_e, sigma_c_1)

    p_sum = p_c_1_e + p_c_0_e
    p_c_0_e, p_c_1_e = p_c_0_e / p_sum, p_c_1_e / p_sum
    x_all.append(e)
    y_0_all.append(p_c_0_e)
    y_1_all.append(p_c_1_e)


plt.plot(x_all, y_0_all, 'k--', label="$p(c=0|M)$")
plt.plot(x_all, y_1_all, 'k', label="$p(c=1|M)$")

plt.plot([mu_c_1] * 100, np.linspace(-0.02, 1.03, 100), label="M=50", color='grey', linestyle='dashdot')

plt.legend()
plt.xlabel("M")
plt.ylabel("$p(c|M)$")
plt.savefig("pcm.eps", format='eps', bbox_inches='tight')
plt.show()