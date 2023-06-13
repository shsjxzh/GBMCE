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


def generate_pic(sigma_s=4, sigma_c_0=10, sigma_c_1=10, mu_c_0=50, mu_c_1=150, left=-30, right=150, color='k'):
    S = []
    ANS = []

    for s in range(left, right):
        p_c_0_s = generate_portion(s, mu_c_0, sigma_s, sigma_c_0)
        p_c_1_s = generate_portion(s, mu_c_1, sigma_s, sigma_c_1)

        p_sum = p_c_1_s + p_c_0_s
        p_c_0_s, p_c_1_s = p_c_0_s / p_sum, p_c_1_s / p_sum

        # E[T|S]
        # c_0:
        e_T_S_C_0 = E_T_S_C(s, sigma_s, mu_c_0, sigma_c_0)
        e_T_S_C_1 = E_T_S_C(s, sigma_s, mu_c_1, sigma_c_1)

        e_T_S_C = p_c_0_s * e_T_S_C_0 + p_c_1_s * e_T_S_C_1

        # E[T|S] - S
        ans = e_T_S_C - s
        S.append(s)
        ANS.append(ans)
    plt.plot(S, ANS, label=r"$\sigma_{c=0}=" + str(sigma_c_1) + r", \sigma_{c=1}=$" +str(sigma_c_0), color=color)

left = -25
right = 200

# set colors
# rocket_palette = sns.color_palette("rocket", as_cmap=True)
Paired_palette = sns.color_palette("Paired")

sigma_c_1_array = [10,30,40,60,200]
for i in range(len(sigma_c_1_array)):
    # color = rocket_palette(i / len(sigma_c_1_array))
    color = Paired_palette[i]
    generate_pic(sigma_c_1=sigma_c_1_array[i], left=left, right=right, color=color)

plt.ylabel('E[T|M] - M')
plt.xlabel('M')
plt.legend()

plt.plot(list(range(left,right)), [0] * (right-left), 'k--')
plt.savefig("fig/egm_m_multi_new.pdf", format='pdf', bbox_inches='tight')
plt.show()