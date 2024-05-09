import numpy as np
import matplotlib.pyplot as plt

# the alpha will be the "famaility" of the items

def E_T_c_M(m, mu_o, sigma_o, sigma_T):
    return (sigma_T ** 2 * mu_o + sigma_o**2 * m) / (sigma_o ** 2 + sigma_T**2)

# p(c|M)
def p_c_M(M, mu_c, sigma_c, sigma_T):
    sigma_2 = sigma_c**2 + sigma_T**2

    p = 1 /np.sqrt(sigma_2) * np.exp(-(M - mu_c) ** 2 / 2 / sigma_2)
    return p

def draw_simu(ax, alpha):
    # simulation objects mean and var in average
    sigma_o = np.sqrt(1/200)
    mu_o = np.linspace(-6.75, 6.75, 10, endpoint=True)
    mu_o = sigma_o * mu_o + 0.5

    sigma_T = np.sqrt(1/12)
    sigma_o_specific = np.sqrt(1/200)
    sigma_T_specific = np.sqrt(1/50)

    # simulation point absolute size:
    xx = 1.5 * np.arange(-3, 4)
    xx = sigma_o * xx + 0.5
    error = np.zeros((4, len(xx)))
    deltas = [2.25 * sigma_o, 0.75 * sigma_o, -0.75 * sigma_o, -2.25 * sigma_o]
    labels = ['very small', 'small', 'large', 'very large']

    count = 0
    for x in xx:
        # not know the category; infer the category

        # now using m
        m = np.random.normal(x, sigma_T, 1000)
        m = m.mean()

        # using only object and category mean / variance
        etm_x = E_T_c_M(m, mu_o=0.5, sigma_o=np.sqrt(1/8.5), sigma_T=sigma_T)

        # etm_x = E_T_c_M

        # know the category
        # try four type: small/very small/large/very large
        for j in range(4):
            delta = deltas[j]
            etm_x_o = E_T_c_M(m, delta+x, sigma_o_specific, sigma_T_specific)

            etm_x_final = etm_x * (1 - alpha) + etm_x_o * alpha
            # error.append(etm_x_final - x)
            error[j,count] = etm_x_final - x
        count += 1

    shape = ['o', '^', 's', 'v']

    if alpha == 0:
        for i in range(4):
            ax.plot(xx, error[i,:], shape[i] + '--', label=labels[i], color='k')
    else:
        for i in range(4):
            ax.plot(xx, error[i,:], shape[i] + '--', color='k')
    # plt.legend()
    ax.title.set_text(r"$\alpha$ = " + str(alpha))
    ax.set_ylim([-0.15, 0.15])
    ax.set_xlim([0,1])

    # once
    if alpha == 0:
        ax.set_ylabel("Remembered - Studied")
    else:
        ax.get_yaxis().set_visible(False)
    ax.set_xlabel("Study Size")

fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
draw_simu(ax1, 0)
draw_simu(ax2, 0.4)
draw_simu(ax3, 0.7)
fig.legend(fancybox=True, framealpha=0.5,loc='upper left', bbox_to_anchor=(0.52,0.88))
plt.savefig("Figure7.pdf",  bbox_inches='tight', format='pdf')
plt.show()









