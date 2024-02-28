import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
from scipy.stats import norm
# from Expection import *

# the alpha will be the "famaility" of the items
import matplotlib

matplotlib.rcParams.update({'font.size': 14})

def E_T_c_M(m, mu_o, sigma_o, sigma_T):
    return (sigma_T ** 2 * mu_o + sigma_o**2 * m) / (sigma_o ** 2 + sigma_T**2)

# p(c|M)
def p_c_M(M, mu_c, sigma_c, sigma_T):
    sigma_2 = sigma_c**2 + sigma_T**2

    p = 1 /np.sqrt(sigma_2) * np.exp(-(M - mu_c) ** 2 / 2 / sigma_2)
    return p

# handling 2 dim cases
sigma_c = [1.2, 1.2]
sigma_T = [1, 1]

sigma_c_2 = [500, 500]
sigma_T_2 = [1, 1]
typical = []
atypical = []

cmp_typical = []
cmp_atypical = []

def calculate_portion(encoded, target):
    error = np.linalg.norm(encoded - target)
    unadjusted_biase = np.linalg.norm(target) - np.linalg.norm(encoded)
    return unadjusted_biase / error

def draw_error_bar(array1, array2, array3, array4):
    mean1 = np.mean(array1)
    std1 = np.std(array1)
    mean2 = np.mean(array2)
    std2 = np.std(array2)
    mean3 = np.mean(array3)
    std3 = np.std(array3)
    mean4 = np.mean(array4)
    std4 = np.std(array4)

    # Data for plotting
    means = [mean1, mean2, mean3, mean4]
    stds = [std1, std2, std3, std4]
    positions = [1, 1.5, 2, 2.5]  # X positions for the plot
    bar_width = 0.3  # Width of the bars

    # Create bar plot
    plt.bar(positions, means, \
        width=bar_width, align='center', alpha=0.5, \
            ecolor='black', capsize=10, edgecolor='black', linewidth=1, \
                 error_kw=dict(lw=1, capsize=5, capthick=1))
    # Adding labels and title
    plt.xticks(positions, ['Typical', 'Atypical', "Scrambled \n'Typical'", "Scrambled \n'Atypical'"])
    plt.ylabel('Avg. Bias (proportion)')
    plt.savefig("Thompson-Schill_graph.pdf",  bbox_inches='tight', format='pdf')
    plt.show()


# create points that have the same distance to the center (0, 0)
center = np.array([0, 0])
center2 = np.array([5, 5])

# Define the radius of the circle
radius = 1

# Generate points
theta = np.linspace(0, 2*np.pi, 30)
x = center[0] + radius * np.cos(theta)
y = center[1] + radius * np.sin(theta)
            
alpha1 = 0.4
alpha2 = 0.6

def generate_error_portion(point, alpha):
    num_m = 7
    # m = np.random.normal(point, sigma_T)
    m = np.random.normal(point, sigma_T, (num_m,2))
    m = m.mean(axis=0)
    # m = point
    # for atypical:
    e_t_m_0 = E_T_c_M(m[0], mu_o=0, sigma_o=sigma_c[0], sigma_T=sigma_T[0])
    e_t_m_1 = E_T_c_M(m[1], mu_o=0, sigma_o=sigma_c[1], sigma_T=sigma_T[1])

    e_t_m = np.array([e_t_m_0, e_t_m_1])

    e_t_m_0_2 = E_T_c_M(m[0], mu_o=center2[0], sigma_o=sigma_c_2[0], sigma_T=sigma_T_2[0])
    e_t_m_1_2 = E_T_c_M(m[1], mu_o=center2[1], sigma_o=sigma_c_2[1], sigma_T=sigma_T_2[1])

    e_t_m_2 = np.array([e_t_m_0_2, e_t_m_1_2])
    return calculate_portion((1-alpha) * e_t_m + alpha * e_t_m_2, point)


for xx in x:
    for yy in y:
        point = np.array([xx, yy])
        for i in range(10):
            typical.append(generate_error_portion(point, alpha1))
            atypical.append(generate_error_portion(point, alpha2))
            cmp_typical.append(generate_error_portion(point, alpha2))
            cmp_atypical.append(generate_error_portion(point, alpha2))

draw_error_bar(typical, atypical, cmp_typical, cmp_atypical)