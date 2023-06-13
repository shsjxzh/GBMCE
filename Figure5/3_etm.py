from cmath import e
from turtle import color
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats


def generate_portion(M, mu_c, sigma_c, sigma_T):
    sigma_2 = sigma_c**2 + sigma_T**2

    portion = 1 /np.sqrt(sigma_2) * np.exp(-(M - mu_c) ** 2 / 2 / sigma_2)
    return portion

def E_T_c_M(m, sigma_T, mu_c, sigma_c):
    return (sigma_T ** 2 * mu_c + (sigma_c**2) * m) / (sigma_T**2 + sigma_c ** 2)

def p_g_c_1(mu_c, sigma_c, m):
    return 1 /sigma_c/ np.sqrt(2) / np.pi * np.exp(-(m - mu_c) ** 2 / 2 / sigma_c**2)


# add another infinite variance category
def generate_pic(ax, sigma_T=4, sigma_c_0=200, sigma_c_1=10, mu_c_0=150, mu_c_1=50, sigma_c_out=400, mu_c_out=150):

    M = []
    ANS = []
    p_c_0 = []
    p_c_1 = []
    p_c_out = []
    pg = []
    pg2 = []
    l = -30
    r = 231
    
    ax.plot(list(range(l,r)), [up] * (r-l), 'k--', label = 'M')
    ax.plot(list(range(l,r)), [0] * (r-l), 'k', label = 'E[T|M]', linestyle="dotted")
    for m in range(l, r):
        p_c_0_m = generate_portion(m, mu_c_0, sigma_c_0, sigma_T)
        p_c_1_m = generate_portion(m, mu_c_1, sigma_c_1, sigma_T)
        p_c_out_m = generate_portion(m, mu_c_1, sigma_c_out, sigma_T)

        p_sum = p_c_1_m + p_c_0_m + p_c_out_m
        p_c_0_m, p_c_1_m, p_c_out_m = p_c_0_m / p_sum, p_c_1_m / p_sum, p_c_out_m / p_sum


        # E[T|M]
        e_T_c_0_M = E_T_c_M(m,  sigma_T, mu_c_0, sigma_c_0)
        e_T_c_1_M = E_T_c_M(m,  sigma_T, mu_c_1, sigma_c_1)
        e_T_c_out_M = E_T_c_M(m,  sigma_T, mu_c_out, sigma_c_out)

        e_T_c_M = e_T_c_0_M * p_c_0_m + e_T_c_1_M * p_c_1_m + e_T_c_out_M * p_c_out_m
    
        ans = e_T_c_M

        M.append(m)
        ANS.append(ans)
        p_c_0.append(p_c_0_m)
        p_c_1.append(p_c_1_m)
        p_c_out.append(p_c_out_m)
        pg.append(p_g_c_1(mu_c_1, sigma_c_1, m))
        pg2.append(p_g_c_1(mu_c_0, sigma_c_0, m))


    ax.plot(M, pg, label=r"$\sigma_{c=1}=" + str(sigma_c_1)+r"$", color='grey')
    ax.plot(M, pg2, label=r"$\sigma_{c=0}=" + str(sigma_c_0)+r"$", color='k')

    for i in range(0, len(M), 10):
        ax.annotate("", xy=(M[i], up), xytext=(ANS[i], 0), arrowprops=dict(arrowstyle="<-", color='darkgrey'))
    
    
    plt.legend()
    plt.savefig("egm_var_{}.eps".format(sigma_c_0), bbox_inches='tight')
    plt.show()
    return M, ANS

up = 0.023

for sigma_c_0 in [400, 30, 10]:
    fig, ax = plt.subplots(1, 1)
    M, ANS = generate_pic(ax, sigma_T=12,sigma_c_0=sigma_c_0, sigma_c_1=10)