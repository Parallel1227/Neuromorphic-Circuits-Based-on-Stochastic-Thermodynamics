# 姓名：彭晓煊
# 时间：2024/6/6 11:10
import matplotlib.pyplot as plt
from pylab import *
import numpy as np
import scipy
from scipy.linalg import null_space
from scipy import signal
from scipy import integrate
from matplotlib import rcParams
import pandas as pd

config = {
    "font.family": 'serif',
    "mathtext.fontset": 'stix',
    "font.serif": ['SimSun'],
}
rcParams.update(config)


def Fermi(x):
    return 1.0 / (exp(x) + 1)


def Bose(x):
    return 1.0 / (exp(x) - 1)


def NEURON(Jin, Vout, Vth, E0, flag, St):
    A = zeros((2, 2))
    Gamma_l = 0.2
    Gamma_r = 0.2
    Gamma_R = 0.002

    kBT = 1.0
    Cg = 200.0

    # if(St == 0):
    # 	Vg = 5.0
    # else:
    # 	if (Vout < Vth):
    # 		Vg = 0.0
    # 	else:
    # 		Vg = 5.0

    if (flag == 0):
        if (Vout < Vth):
            Vg = 0.0
        else:
            Vg = E0
            flag = 1
    else:
        if (Vout <= (0.001)):
            Vg = 0.0
            flag = 0
        else:
            Vg = E0

    if (St == 0):
        Vg = E0

    # print(flag, Vg, Vout)

    E_N = E0 - Vg

    mu_l = 0.0
    mu_r = mu_l - Vout

    k_Nl = Gamma_l * Fermi((E_N - mu_l) / kBT)
    k_lN = Gamma_l * (1.0 - Fermi((E_N - mu_l) / kBT))
    k_rN = Gamma_r * (1.0 - Fermi((E_N - mu_r) / kBT))
    k_Nr = Gamma_r * Fermi((E_N - mu_r) / kBT)

    A[1][0] = k_Nr + k_Nl
    A[0][0] = -A[1][0]
    A[0][1] = k_rN + k_lN
    A[1][1] = -A[0][1]

    p = null_space(A)
    sum = p[0][0] + p[1][0]
    p[0][0] = p[0][0] / sum
    p[1][0] = p[1][0] / sum
    p_N = p[1][0]
    J_d = k_rN * p_N - k_Nr * (1 - p_N)
    J_GND = 0.5 * Gamma_R * (Fermi((mu_l - mu_l) / kBT) - Fermi((mu_l - mu_r) / kBT))
    # 电压控制的开关 放电时屏蔽输入电流
    if (flag == 0):
        Vout += 1.0 * (Jin - J_d - J_GND) * tint / Cg
    else:
        Vout += 1.0 * (0.0 - J_d - J_GND) * tint / Cg
        # diss_MOS = 1.0 * J_d * tint * (mu_l - mu_r)
        # diss_GND = J_GND * J_GND * 8.432 / Gamma_R
    diss_MOS = 1.0 * J_d * tint * (mu_l - mu_r)
    diss_GND = 1.0 * J_GND * tint * (mu_l - mu_r)

    return J_d, J_GND, Vout, Vg, flag, diss_MOS, diss_GND

##################################表格数据1####################################
diss_MOS_R = np.zeros(51)
diss_GND_R = np.zeros(51)
delta_output = np.zeros(51)
diss_MOS_output = np.zeros(51)
diss_GND_output = np.zeros(51)
Jin2_output = np.zeros(51)
j = 0

# Vout_E = 0.0
# Vout_I = 0.0
# # Jin = 0.01 # r到N
E0 = 5.0

tint = 10
T = 1000000
Ntot = int(T / tint)
Vth = 5.0
# flag_E = 0
# flag_I = 0


output_E = np.zeros(Ntot)
output_I = np.zeros(Ntot)
J_E = np.zeros(Ntot)
# Jin1 = np.zeros(Ntot)
# Jin2 = np.zeros(Ntot)
Vg_E = np.zeros(Ntot)
time = np.zeros(Ntot)
diss_MOS = np.zeros(Ntot)
flag = np.zeros(Ntot)
energy_MOS = np.zeros(Ntot)
diss_GND = np.zeros(Ntot)
energy_GND = np.zeros(Ntot)

for delta_t in range(0, 5001, 200):
    print(delta_t)
    delta_output[j] = delta_t * tint
    t_S1 = 0
    S1 = 1100
    S2 = 1100
    t_S2 = t_S1 + delta_t

    J_GND = np.zeros(Ntot)
    Jin1 = np.zeros(Ntot)
    Jin2 = np.zeros(Ntot)
    Jin1[t_S1:t_S1 + S1] = 0.05
    Jin2[t_S2:t_S2 + S2] = 0.05
    flag_output = 0
    flag_E = 0
    Vout_E = 0.0
    diss_MOS_R[j] = 0.0
    diss_GND_R[j] = 0.0
    for i in range(Ntot):
        output_E[i] = Vout_E
        J_E[i], J_GND[i], Vout_E, Vg_E_R, flag_E, diss_MOS[i], diss_GND[i] = NEURON(Jin1[i] + Jin2[i], Vout_E, Vth, E0, flag_E, 1)
        # J_I[i], Vout_I, Vg_I_R, flag_I = NEURON(Jin[i], Vout_I, Vth, E0, flag_I, 0)
        Vg_E[i] = Vg_E_R
        flag[i] = flag_E
        diss_MOS_R[j] = diss_MOS[i] + diss_MOS_R[j]
        energy_MOS[i] = diss_MOS_R[j]
        diss_GND_R[j] = diss_GND[i] + diss_GND_R[j]
        energy_GND[i] = diss_GND_R[j]
        time[i] = i * tint
    j = j + 1

data = {'delta_t': delta_output, 'diss': diss_MOS_R + diss_GND_R, 'diss_MOS': diss_MOS_R, 'diss_GND': diss_GND_R}
df = pd.DataFrame(data)
df.to_excel('output.xlsx', index=False)
##################################表格数据1####################################

##################################表格数据2####################################
# diss_MOS_R = np.zeros(37)
# diss_GND_R = np.zeros(37)
# delta_output = np.zeros(37)
# diss_MOS_output = np.zeros(37)
# diss_GND_output = np.zeros(37)
# Jin2_output = np.zeros(37)
# j = 0
#
# # Vout_E = 0.0
# # Vout_I = 0.0
# # # Jin = 0.01 # r到N
# E0 = 5.0
#
# tint = 10
# T = 1000000
# Ntot = int(T / tint)
# Vth = 5.0
# # flag_E = 0
# # flag_I = 0
#
#
# output_E = np.zeros(Ntot)
# output_I = np.zeros(Ntot)
# J_E = np.zeros(Ntot)
# # Jin1 = np.zeros(Ntot)
# # Jin2 = np.zeros(Ntot)
# Vg_E = np.zeros(Ntot)
# time = np.zeros(Ntot)
# diss_MOS = np.zeros(Ntot)
# flag = np.zeros(Ntot)
# energy_MOS = np.zeros(Ntot)
# diss_GND = np.zeros(Ntot)
# energy_GND = np.zeros(Ntot)
# Jin_start = 0.013
#
# for delta_t in range(2000, 82001, 5000):
#     print(delta_t)
#     delta_output[j] = delta_t
#     t_S1 = 0
#     S1 = 1500
#     S2 = 1500
#     t_S2 = t_S1 + delta_t
#
#     for Jin in np.arange(Jin_start, 0.5, 0.0001):
#         print('Jin = ', Jin)
#         J_GND = np.zeros(Ntot)
#         Jin1 = np.zeros(Ntot)
#         Jin2 = np.zeros(Ntot)
#         Jin1[t_S1:t_S1 + S1] = 0.05
#         Jin2[t_S2:t_S2 + S2] = Jin
#         flag_output = 0
#         flag_E = 0
#         Vout_E = 0.0
#         diss_MOS_R[j] = 0.0
#         diss_GND_R[j] = 0.0
#         for i in range(Ntot):
#             output_E[i] = Vout_E
#             J_E[i], J_GND[i], Vout_E, Vg_E_R, flag_E, diss_MOS[i], diss_GND[i] = NEURON(Jin1[i] + Jin2[i], Vout_E, Vth, E0, flag_E, 1)
#             # J_I[i], Vout_I, Vg_I_R, flag_I = NEURON(Jin[i], Vout_I, Vth, E0, flag_I, 0)
#             Vg_E[i] = Vg_E_R
#             flag[i] = flag_E
#             diss_MOS_R[j] = diss_MOS[i] + diss_MOS_R[j]
#             energy_MOS[i] = diss_MOS_R[j]
#             diss_GND_R[j] = diss_GND[i] + diss_GND_R[j]
#             energy_GND[i] = diss_GND_R[j]
#             time[i] = i * tint
#             if(flag_E == 1):
#                 flag_output = 1
#         print('flag_output = ', flag_output)
#         if(flag_output == 1):
#             Jin2_output[j] = Jin
#             diss_MOS_output[j] = diss_MOS_R[j]
#             diss_GND_output[j] = diss_GND_R[j]
#             Jin_start = Jin
#             print(delta_output[j], Jin2_output[j], diss_MOS_output[j], diss_GND_output[j])
#             break
#
#     j = j + 1
#
# data = {'delta_t': delta_output, 'Jin2': Jin2_output, 'diss_MOS': diss_MOS_output, 'diss_GND': diss_GND_output}
# df = pd.DataFrame(data)
# df.to_excel('output.xlsx', index=False)
##################################表格数据2####################################

##################################test####################################
# Vout_E = 0.0
# Vout_I = 0.0
# # Jin = 0.01 # r到N
# E0 = 5.0
#
# tint = 10
# T = 1000000
# Ntot = int(T / tint)
# Vth = 5.0
# flag_E = 0
# flag_I = 0
#
# output_E = np.zeros(Ntot)
# output_I = np.zeros(Ntot)
# J_E = np.zeros(Ntot)
# Jin1 = np.zeros(Ntot)
# Jin2 = np.zeros(Ntot)
# Vg_E = np.zeros(Ntot)
# time = np.zeros(Ntot)
# diss_MOS = np.zeros(Ntot)
# flag = np.zeros(Ntot)
# energy_MOS = np.zeros(Ntot)
# diss_GND = np.zeros(Ntot)
# energy_GND = np.zeros(Ntot)
# diss_MOS_R = 0.0
# diss_GND_R = 0.0
# delta_t = 30000
# t_S1 = 0
# S1 = 2000
# S2 = 2000
# t_S2 = t_S1 + delta_t
#
# Jin = np.zeros(Ntot)
# J_GND = np.zeros(Ntot)
# Jin1[t_S1:t_S1 + S1] = 0.05
# Jin2[t_S2:t_S2 + S2] = 0.05
#
# for i in range(Ntot):
#     output_E[i] = Vout_E
#     output_I[i] = Vout_I
#     J_E[i], J_GND[i], Vout_E, Vg_E_R, flag_E, diss_MOS[i], diss_GND[i] = NEURON(Jin1[i] + Jin2[i], Vout_E, Vth, E0, flag_E, 1)
#     # J_I[i], Vout_I, Vg_I_R, flag_I = NEURON(Jin[i], Vout_I, Vth, E0, flag_I, 0)
#     Vg_E[i] = Vg_E_R
#     flag[i] = flag_E
#     diss_MOS_R = diss_MOS[i] + diss_MOS_R
#     energy_MOS[i] = diss_MOS_R
#     diss_GND_R = diss_GND[i] + diss_GND_R
#     energy_GND[i] = diss_GND_R
#     time[i] = i * tint
#
# print(diss_MOS_R, diss_GND_R)
#
#
# plt.figure(1, figsize=(15, 10), dpi=300)
# plt.rcParams['axes.formatter.min_exponent'] = 1  # 将最小指数更改为1
# plt.rc('font', size=16, family='Times New Roman')  # 将字体大小更改为12
# plt.rcParams['xtick.direction'] = 'in'  # 将x周的刻度线方向设置向内
# plt.rcParams['ytick.direction'] = 'in'  # 将y轴的刻度方向设置向内
# plt.subplot(2, 2, 1)
# plt.plot(time, Jin1, color='orange')
# plt.plot(time, Jin2, color='cornflowerblue')
# plt.xlabel("Time")
# plt.ylabel("Jin")
# plt.subplot(2, 2, 2)
# plt.plot(time, output_E, color='orange')
# # plt.plot(time, output_I, color='cornflowerblue')
# plt.xlabel("Time")
# plt.ylabel("Vout")
# plt.subplot(2, 2, 3)
# plt.plot(time, J_E, color='orange')
# plt.plot(time, J_GND, color='cornflowerblue')
# plt.xlabel("Time")
# plt.ylabel("J")
# plt.subplot(2, 2, 4)
# plt.plot(time, energy_MOS, color='orange')
# plt.plot(time, energy_GND, color='cornflowerblue')
# plt.plot(time, energy_MOS+energy_GND, color='black')
# plt.ylabel("diss")
# # plt.plot(time, flag, color='orange')
# # plt.xlabel("Time")
# # plt.ylabel("flag")
#
# plt.tight_layout()
# plt.savefig('test.svg', format='svg')
##################################test####################################