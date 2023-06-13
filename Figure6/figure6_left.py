import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def generate_portion(s, mu_c, sigma_s, sigma_c):
    a = 1 / 2 / sigma_s ** 2 + 1 / 2 / sigma_c ** 2
    b = - s / sigma_s ** 2 - mu_c / sigma_c ** 2
    c = s ** 2 / 2 / sigma_s ** 2 + mu_c ** 2 / 2 / sigma_c ** 2

    portion = np.exp(b ** 2 / 4 / a - c) / np.sqrt(a) / sigma_s ** 2 / sigma_c ** 2
    return portion

def E_T_S_C(s,  sigm_s, mu_c, sigma_c):
    return (sigma_c ** 2 * s + sigm_s ** 2 * mu_c) / (sigm_s ** 2 + sigma_c ** 2)


def generate_pic(sigma_s=4, sigma_c_0=10, sigma_c_1=10, mu_c_0=150, mu_c_1=50, left=-30, right=150, color='k'):
    S = []
    ANS = []

    for s in range(left, right):
        p_c_0_s = generate_portion(s, mu_c_0, sigma_s, sigma_c_0)
        p_c_1_s = generate_portion(s, mu_c_1, sigma_s, sigma_c_1)

        p_sum = p_c_1_s + p_c_0_s
        p_c_0_s, p_c_1_s = p_c_0_s / p_sum, p_c_1_s / p_sum

        S.append(s)
        ANS.append(p_c_1_s)
    plt.plot(S, ANS, label=r"$\sigma_{c=0}=" + str(sigma_c_0) + r", \sigma_{c=1}=$" +str(sigma_c_1), color=color)

left = -100
right = 175

# set colors
rocket_palette = sns.color_palette("rocket", as_cmap=True)
# Paired_palette = sns.color_palette("Paired")


sigma_c_0_array = [10,20,30,40,70,200]
for i in range(len(sigma_c_0_array)):
# for i in range(10, 31, 20):
    color = rocket_palette(i / len(sigma_c_0_array))
    # color = Paired_palette[i]
    generate_pic(sigma_c_0=sigma_c_0_array[i], left=left, right=right, color=color)

plt.ylabel('p(c=1|M)')
plt.xlabel('M')

plt.plot([50] * 100, np.linspace(-0.02, 1.03, 100), "k--", label="M=50")

plt.legend()
plt.savefig("fig/pcm_multi.pdf", format='pdf', bbox_inches='tight')
plt.show()