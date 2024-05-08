import numpy as np
import matplotlib.pyplot as plt

import matplotlib
matplotlib.rcParams.update({'font.size': 18})

np.random.seed(seed=0)

def generate_portion(s, mu_c, sigma_s, sigma_c):
    a = 1 / 2 / sigma_s ** 2 + 1 / 2 / sigma_c ** 2
    b = - s / sigma_s ** 2 - mu_c / sigma_c ** 2
    c = s ** 2 / 2 / sigma_s ** 2 + mu_c ** 2 / 2 / sigma_c ** 2

    portion = np.exp(b ** 2 / 4 / a - c) / np.sqrt(a) / sigma_s ** 2 / sigma_c ** 2
    return portion

def E_T_S_C(s,  sigm_s, mu_c, sigma_c):
    return (sigma_c ** 2 * s + sigm_s ** 2 * mu_c) / (sigm_s ** 2 + sigma_c ** 2)


def generate_pic(sigma_s=6, sigma_c_0=10, sigma_c_1=10, mu_c_0=150, mu_c_1=50):
    S = []
    ANS = []

    m_std = []

    # now, s is the ground truth value
    # noisy memoroy m is sampled from s
    for s in range(-10, 110):

        m = np.random.normal(s, sigma_s, 1000)
        m_std.append(np.std(m))
        m = m.mean()

        # p_c_0_s = generate_portion(m, mu_c_0, sigma_s, sigma_c_0)
        # p_c_1_s = generate_portion(m, mu_c_1, sigma_s, sigma_c_1)

        # p_sum = p_c_1_s + p_c_0_s
        # p_c_0_s, p_c_1_s = p_c_0_s / p_sum, p_c_1_s / p_sum

        # E[T|S]
        # c_0:
        # e_T_S_C_0 = E_T_S_C(m, sigma_s, mu_c_0, sigma_c_0)
        e_T_S_C_1 = E_T_S_C(m, sigma_s, mu_c_1, sigma_c_1)

        # e_T_S_C = p_c_0_s * e_T_S_C_0 + p_c_1_s * e_T_S_C_1
        e_T_S_C = e_T_S_C_1

        # E[T|S] - S
        ans = e_T_S_C - s
        S.append(s)
        ANS.append(ans)
    plt.plot(S, ANS, 'k')

    ANS = np.array(ANS)


plt.plot(list(range(-10,110)), [0] * 120, 'k--')
generate_pic(sigma_c_0=200)
plt.ylabel('E[T|M] - T')
plt.xlabel('T')
# plt.savefig("egm_t_cam.png", format='png', bbox_inches='tight')
plt.savefig("Figure4B.pdf", format='pdf', bbox_inches='tight')
plt.show()