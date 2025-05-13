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
import openpyxl

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


def NEURON(Jin, Vout, Vth, E0, flag, St, delta_E0):
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

    E_N = E0 - Vg + delta_E0

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


##########################################test############################################
# Vout_E = 0.0
# Vout_I = 0.0
# E0 = 5.0
# delta_E0 = -0.0
#
# tint = 10
# T = 1000000
# Ntot = int(T / tint)
# t = np.linspace(0, T, tint)
# Vth = 5.0
# flag_E = 0
# flag_break = 0
#
# output_E = np.zeros(Ntot)
# output_I = np.zeros(Ntot)
# Jout = np.zeros(Ntot)
# Vg_E = np.zeros(Ntot)
# time = np.zeros(Ntot)
# flag = np.zeros(Ntot)
# diss_MOS = np.zeros(Ntot)
# energy_MOS = np.zeros(Ntot)
# diss_GND = np.zeros(Ntot)
# energy_GND = np.zeros(Ntot)
# diss_MOS_R = 0.0
# diss_GND_R = 0.0
#
# Jin = np.zeros(Ntot)
# J_GND = np.zeros(Ntot)
# start_t = 20000
# delta_t = 5636
# Jin[start_t:start_t + delta_t] = 0.05
# for i in range(Ntot):
#     output_E[i] = Vout_E
#     output_I[i] = Vout_I
#     Jout[i], J_GND[i], Vout_E, Vg_E_R, flag_E, diss_MOS[i], diss_GND[i] = NEURON(Jin[i], Vout_E, Vth, E0, flag_E, 1, delta_E0)
#     Vg_E[i] = Vg_E_R
#     flag[i] = flag_E
#     diss_MOS_R = diss_MOS[i] + diss_MOS_R
#     energy_MOS[i] = diss_MOS_R
#     diss_GND_R = diss_GND[i] + diss_GND_R
#     energy_GND[i] = diss_GND_R
#     time[i] = i * tint
#     if(i > 0):
#         if(output_E[i-1] >= Vth * 0.9)&(output_E[i] < Vth * 0.9):
#             print(i-1-start_t, output_E[i-1])
#
# print(diss_MOS_R, diss_GND_R, diss_MOS_R + diss_GND_R)
#
# plt.figure(1, figsize=(15, 10), dpi=300)
# plt.rcParams['axes.formatter.min_exponent'] = 1  # 将最小指数更改为1
# plt.rc('font', size=16, family='Times New Roman')  # 将字体大小更改为12
# plt.rcParams['xtick.direction'] = 'in'  # 将x周的刻度线方向设置向内
# plt.rcParams['ytick.direction'] = 'in'  # 将y轴的刻度方向设置向内
# plt.subplot(2, 2, 1)
# plt.plot(time, Jin, color='orange')
# plt.xlabel("Time")
# plt.ylabel("Jin")
# plt.subplot(2, 2, 2)
# plt.plot(time, output_E, color='orange')
# # plt.plot(time, output_I, color='cornflowerblue')
# plt.xlabel("Time")
# plt.ylabel("Vout")
# plt.subplot(2, 2, 3)
# plt.plot(time, Jout, color='orange')
# plt.plot(time, J_GND, color='cornflowerblue')
# print(max(output_E),np.argmax(output_E),output_E[np.argmax(output_E)+1])
#
# plt.xlabel("Time")
# plt.ylabel("J")
# plt.subplot(2, 2, 4)
# plt.plot(time, energy_MOS, color='orange')
# plt.plot(time, energy_GND, color='cornflowerblue')
# plt.plot(time, energy_MOS + energy_GND, color='black')
# plt.xlabel("Time")
# plt.ylabel("diss")
#
# plt.tight_layout()
# plt.savefig('test.svg', format='svg')

##########################################test############################################

##########################################兴奋/抑制_调控延时############################################
# Vout_E = 0.0
# Vout_I = 0.0
# E0 = 5.0
# # delta_E0 = -2.5
#
# tint = 10
# T = 1000000
# Ntot = int(T / tint)
# t = np.linspace(0, T, tint)
# Vth = 5.0
# # flag_E = 0
# # flag_break = 0
# #
# # output_E = np.zeros(Ntot)
# # output_I = np.zeros(Ntot)
# # Jout = np.zeros(Ntot)
# # Vg_E = np.zeros(Ntot)
# # time = np.zeros(Ntot)
# # flag = np.zeros(Ntot)
# # diss_MOS = np.zeros(Ntot)
# # energy_MOS = np.zeros(Ntot)
# # diss_GND = np.zeros(Ntot)
# # energy_GND = np.zeros(Ntot)
# # diss_MOS_R = 0.0
# # diss_GND_R = 0.0
#
# Jin = np.zeros(Ntot)
# Jout_max = np.zeros(20)
# delta = np.zeros(20)
# energy_sum = np.zeros(20)
# delta_E0 = np.zeros(20)
# k = 0
# J_GND = np.zeros(Ntot)
# start_t = 20000
# delta_t = 5636
# Jin[start_t:start_t + delta_t] = 0.05
# for j in np.arange(-4.5,5.1,0.5):
#     delta_E0[k] = j
#     flag_E = 0
#     flag_break = 0
#
#     output_E = np.zeros(Ntot)
#     output_I = np.zeros(Ntot)
#     Jout = np.zeros(Ntot)
#     Vg_E = np.zeros(Ntot)
#     time = np.zeros(Ntot)
#     flag = np.zeros(Ntot)
#     diss_MOS = np.zeros(Ntot)
#     energy_MOS = np.zeros(Ntot)
#     diss_GND = np.zeros(Ntot)
#     energy_GND = np.zeros(Ntot)
#     Vout_E = 0.0
#     diss_MOS_R = 0.0
#     diss_GND_R = 0.0
#     for i in range(Ntot):
#         output_E[i] = Vout_E
#         output_I[i] = Vout_I
#         Jout[i], J_GND[i], Vout_E, Vg_E_R, flag_E, diss_MOS[i], diss_GND[i] = NEURON(Jin[i], Vout_E, Vth, E0, flag_E, 1, delta_E0[k])
#         Vg_E[i] = Vg_E_R
#         flag[i] = flag_E
#         diss_MOS_R = diss_MOS[i] + diss_MOS_R
#         energy_MOS[i] = diss_MOS_R
#         diss_GND_R = diss_GND[i] + diss_GND_R
#         energy_GND[i] = diss_GND_R
#         time[i] = i * tint
#     Jout_max[k] = max(Jout)
#     delta[k] = np.argmax(output_E) - 22037
#     energy_sum[k] = diss_MOS_R + diss_GND_R
#     k = k + 1
#     print(k)
#
# # print(diss_MOS_R, diss_GND_R, diss_MOS_R + diss_GND_R)
#
# plt.figure(1, figsize=(15, 7), dpi=300)
# plt.rcParams['axes.formatter.min_exponent'] = 1  # 将最小指数更改为1
# plt.rc('font', size=16, family='Times New Roman')  # 将字体大小更改为12
# plt.rcParams['xtick.direction'] = 'in'  # 将x周的刻度线方向设置向内
# plt.rcParams['ytick.direction'] = 'in'  # 将y轴的刻度方向设置向内
# plt.plot(delta_E0, delta, color='orange')
# plt.xlabel("delta_E0")
# plt.ylabel("delta")
# # plt.subplot(2, 2, 1)
# # plt.plot(time, Jin, color='orange')
# # plt.xlabel("Time")
# # plt.ylabel("Jin")
# # plt.subplot(2, 2, 2)
# # plt.plot(time, output_E, color='orange')
# # # plt.plot(time, output_I, color='cornflowerblue')
# # plt.xlabel("Time")
# # plt.ylabel("Vout")
# # plt.subplot(2, 2, 3)
# # plt.plot(time, Jout, color='orange')
# # plt.plot(time, J_GND, color='cornflowerblue')
# # print(max(Jout))
# # plt.xlabel("Time")
# # plt.ylabel("J")
# # plt.subplot(2, 2, 4)
# # plt.plot(time, energy_MOS, color='orange')
# # plt.plot(time, energy_GND, color='cornflowerblue')
# # plt.plot(time, energy_MOS + energy_GND, color='black')
# # plt.xlabel("Time")
# # plt.ylabel("diss")
#
# plt.tight_layout()
# plt.savefig('兴奋抑制_定值.svg', format='svg')
# data = {'delta_E0': delta_E0, 'Jout_max': Jout_max, 'delta': delta, 'energy_sum': energy_sum}
# df = pd.DataFrame(data)
# df.to_excel('output.xlsx', index=False)

##########################################兴奋/抑制_调控延时############################################

##########################################XOR_test############################################
# Vout_E1 = 0.0
# Vout_E2 = 0.0
# E0 = 1.0
# # delta_E0 = -1.0
#
# tint = 10
# T = 1000000
# Ntot = int(T / tint)
# t = np.linspace(0, T, tint)
# Vth = 1.0
# flag_E1 = 0
# flag_E2 = 0
# flag_firsttime = 1
# i_begin = 0
#
# output_E1 = np.zeros(Ntot)
# output_E2 = np.zeros(Ntot)
# Jout = np.zeros(Ntot)
# Vg_E = np.zeros(Ntot)
# time = np.zeros(Ntot)
# flag = np.zeros(Ntot)
# diss_MOS1 = np.zeros(Ntot)
# energy_MOS1 = np.zeros(Ntot)
# diss_GND1 = np.zeros(Ntot)
# energy_GND1 = np.zeros(Ntot)
# diss_MOS_R1 = 0.0
# diss_GND_R1 = 0.0
# diss_MOS2 = np.zeros(Ntot)
# energy_MOS2 = np.zeros(Ntot)
# diss_GND2 = np.zeros(Ntot)
# energy_GND2 = np.zeros(Ntot)
# diss_MOS_R2 = 0.0
# diss_GND_R2 = 0.0
# Jout_post = np.zeros(Ntot)
#
# Jin1_1 = np.zeros(Ntot)
# Jin1_2 = np.zeros(Ntot)
# Jin2_1 = np.zeros(Ntot)
# Jin2_2 = np.zeros(Ntot)
# J_GND = np.zeros(Ntot)
# start_t = 20000
# delta_t = 3000
# Jin1_1[19992:20478] = 0.05    #控制
# Jin1_2[19992:20478] = 0.0    #控制
# Jin2_1[20000:20486] = 0.05    #输入输出
# Jin2_2[20000:20486] = 0.0    #输入输出
# for i in range(Ntot):
#     output_E1[i] = Vout_E1
#     output_E2[i] = Vout_E2
#     _, _, Vout_E1, _, flag_E1, diss_MOS1[i], diss_GND1[i] = NEURON(Jin1_1[i] + Jin1_2[i], Vout_E1, Vth, E0, flag_E1, 1, -2.0)
#     Jout[i], J_GND[i], Vout_E2, Vg_E_R, flag_E2, diss_MOS2[i], diss_GND2[i] = NEURON(Jin2_1[i] + Jin2_2[i], Vout_E2, Vth, E0, flag_E2, 1, 0.0)
#     # Vg_E[i] = Vg_E_R
#     flag[i] = flag_E2
#     diss_MOS_R1 = diss_MOS1[i] + diss_MOS_R1
#     energy_MOS1[i] = diss_MOS_R1
#     diss_GND_R1 = diss_GND1[i] + diss_GND_R1
#     energy_GND1[i] = diss_GND_R1
#     diss_MOS_R2 = diss_MOS2[i] + diss_MOS_R2
#     energy_MOS2[i] = diss_MOS_R2
#     diss_GND_R2 = diss_GND2[i] + diss_GND_R2
#     energy_GND2[i] = diss_GND_R2
#     time[i] = i * tint
#     if(i > 0):
#         if(flag[i-1] == 0)&(flag[i] == 1):
#             i_begin = i
#         if(flag[i] == 1)&(output_E2[i] > output_E1[i])&(flag_firsttime == 1)&(i - i_begin < 486):
#             Jout_post[i] = 0.05
#         if(Jout_post[i-1] != 0)&(Jout_post[i] == 0):
#             flag_firsttime = 0
#             print(i - i_begin)
#
# print(diss_MOS_R1 + diss_GND_R1, diss_MOS_R2 + diss_GND_R2, diss_MOS_R1 + diss_GND_R1 + diss_MOS_R2 + diss_GND_R2)
#
# plt.figure(1, figsize=(15, 10), dpi=300)
# plt.rcParams['axes.formatter.min_exponent'] = 1  # 将最小指数更改为1
# plt.rc('font', size=16, family='Times New Roman')  # 将字体大小更改为12
# plt.rcParams['xtick.direction'] = 'in'  # 将x周的刻度线方向设置向内
# plt.rcParams['ytick.direction'] = 'in'  # 将y轴的刻度方向设置向内
# plt.subplot(2, 2, 1)
# plt.plot(time, Jin1_1 + Jin1_2, color='cornflowerblue')
# plt.plot(time, Jin2_1 + Jin2_2, color='orange')
# plt.xlabel("Time")
# plt.ylabel("Jin")
# plt.subplot(2, 2, 2)
# plt.plot(time, output_E1, color='cornflowerblue')
# plt.plot(time, output_E2, color='orange')
# plt.xlabel("Time")
# plt.ylabel("Vout")
# plt.subplot(2, 2, 3)
# plt.plot(time, Jout_post, color='orange')
# # plt.plot(time, J_GND, color='cornflowerblue')
# # print(max(Jout))
# plt.xlabel("Time")
# plt.ylabel("Jout_post")
# plt.subplot(2, 2, 4)
# plt.plot(time, energy_MOS1 + energy_MOS2, color='orange')
# plt.plot(time, energy_GND1 + energy_GND2, color='cornflowerblue')
# plt.plot(time, energy_MOS1 + energy_GND1 + energy_MOS2 + energy_GND2, color='black')
# plt.xlabel("Time")
# plt.ylabel("diss")
#
# plt.tight_layout()
# plt.savefig('test.svg', format='svg')

##########################################XOR_test############################################

##########################################XOR_绘图############################################

Vout_E1 = 0.0
Vout_E2 = 0.0
E0 = 1.0
# delta_E0 = -1.0

tint = 10
T = 1000000
Ntot = int(T / tint)
t = np.linspace(0, T, tint)
Vth = 1.0
flag_E1 = 0
flag_E2 = 0
flag_firsttime = 1
i_begin = 0

output_E1 = np.zeros(Ntot)
output_E2 = np.zeros(Ntot)
Jout = np.zeros(Ntot)
Vg_E = np.zeros(Ntot)
time = np.zeros(Ntot)
flag = np.zeros(Ntot)
diss_MOS1 = np.zeros(Ntot)
energy_MOS1 = np.zeros(Ntot)
diss_GND1 = np.zeros(Ntot)
energy_GND1 = np.zeros(Ntot)
diss_MOS_R1 = 0.0
diss_GND_R1 = 0.0
diss_MOS2 = np.zeros(Ntot)
energy_MOS2 = np.zeros(Ntot)
diss_GND2 = np.zeros(Ntot)
energy_GND2 = np.zeros(Ntot)
diss_MOS_R2 = 0.0
diss_GND_R2 = 0.0
Jout_post = np.zeros(Ntot)

Jin1_1 = np.zeros(Ntot)
Jin1_2 = np.zeros(Ntot)
Jin2_1 = np.zeros(Ntot)
Jin2_2 = np.zeros(Ntot)
J_GND = np.zeros(Ntot)
start_t = 20000
delta_t = 3000
# Jin1_1[19992:20478] = 0.00    #控制
# Jin1_2[19992:20478] = 0.00    #控制
# Jin2_1[20000:20486] = 0.00    #输入输出
# Jin2_2[20000:20486] = 0.00    #输入输出
# Jin1_1[39992:40478] = 0.05    #控制
# Jin1_2[39992:40478] = 0.0    #控制
# Jin2_1[40000:40486] = 0.05    #输入输出
# Jin2_2[40000:40486] = 0.0    #输入输出
# Jin1_1[59992:60478] = 0.0    #控制
# Jin1_2[59992:60478] = 0.05    #控制
# Jin2_1[60000:60486] = 0.0    #输入输出
# Jin2_2[60000:60486] = 0.05    #输入输出
# Jin1_1[79992:80478] = 0.05    #控制
# Jin1_2[79992:80478] = 0.05    #控制
# Jin2_1[80000:80486] = 0.05    #输入输出
# Jin2_2[80000:80486] = 0.05    #输入输出
Jin1_1[20000:20243] = 0.00    #控制
Jin1_2[20000:20243] = 0.00    #控制
Jin2_1[20000:20486] = 0.00    #输入输出
Jin2_2[20000:20486] = 0.00    #输入输出
Jin1_1[40000:40243] = 0.05    #控制
Jin1_2[40000:40243] = 0.0    #控制
Jin2_1[40000:40486] = 0.05    #输入输出
Jin2_2[40000:40486] = 0.0    #输入输出
Jin1_1[60000:60243] = 0.0    #控制
Jin1_2[60000:60243] = 0.05    #控制
Jin2_1[60000:60486] = 0.0    #输入输出
Jin2_2[60000:60486] = 0.05    #输入输出
Jin1_1[80000:80243] = 0.05    #控制
Jin1_2[80000:80243] = 0.05    #控制
Jin2_1[80000:80486] = 0.05    #输入输出
Jin2_2[80000:80486] = 0.05    #输入输出

for i in range(Ntot):
    output_E1[i] = Vout_E1
    output_E2[i] = Vout_E2
    _, _, Vout_E1, _, flag_E1, diss_MOS1[i], diss_GND1[i] = NEURON(Jin1_1[i] + Jin1_2[i], Vout_E1, Vth, E0, flag_E1, 1, 0.0)
    Jout[i], J_GND[i], Vout_E2, Vg_E_R, flag_E2, diss_MOS2[i], diss_GND2[i] = NEURON(Jin2_1[i] + Jin2_2[i], Vout_E2, Vth, E0, flag_E2, 1, 0.0)
    # Vg_E[i] = Vg_E_R
    flag[i] = flag_E2
    diss_MOS_R1 = diss_MOS1[i] + diss_MOS_R1
    energy_MOS1[i] = diss_MOS_R1
    diss_GND_R1 = diss_GND1[i] + diss_GND_R1
    energy_GND1[i] = diss_GND_R1
    diss_MOS_R2 = diss_MOS2[i] + diss_MOS_R2
    energy_MOS2[i] = diss_MOS_R2
    diss_GND_R2 = diss_GND2[i] + diss_GND_R2
    energy_GND2[i] = diss_GND_R2
    time[i] = i * tint
    if(i > 0):
        if(flag[i-1] == 0)&(flag[i] == 1):
            i_begin = i
        if(flag[i] == 1)&(output_E2[i] >= output_E1[i])&(i - i_begin < 49):
            Jout_post[i] = 0.05
            print(i - i_begin)

print(diss_MOS_R1 + diss_GND_R1, diss_MOS_R2 + diss_GND_R2, diss_MOS_R1 + diss_GND_R1 + diss_MOS_R2 + diss_GND_R2)

plt.figure(1, figsize=(10, 8), dpi=300)
plt.rcParams['axes.formatter.min_exponent'] = 1  # 将最小指数更改为1
plt.rc('font', size=14, family='Times New Roman')  # 将字体大小更改为16
plt.rcParams['xtick.direction'] = 'in'  # 将x周的刻度线方向设置向内
plt.rcParams['ytick.direction'] = 'in'  # 将y轴的刻度方向设置向内

subplot(4, 1, 1)
plot(time, Jin2_1, color='#4c6792', linewidth='2')
plt.xlabel("Time ($β\hbar$)")
plt.ylabel("Jin1")
# plt.text(-0.15e+7, 5, '(a)', fontsize=16, fontproperties='Times New Roman')

subplot(4, 1, 2)
plot(time, Jin2_2, color='#4c6792', linewidth='2')
plt.xlabel("Time ($β\hbar$)")
plt.ylabel("Jin2")
# plt.text(-0.15e+7, 5, '(b)', fontsize=16, fontproperties='Times New Roman')

subplot(4, 1, 3)
plot(time, output_E1, color='#4c6792', linewidth='2')
plot(time, output_E2, color='orange', linewidth='2')
plt.xlabel("Time ($β\hbar$)")
plt.ylabel("Vout")
# plt.text(-0.15e+7, 4.5, '(c)', fontsize=16, fontproperties='Times New Roman')

subplot(4, 1, 4)
plot(time, Jout_post, color='#4c6792', linewidth='2')
plt.xlabel("Time ($β\hbar$)")
plt.ylabel("Jout_post")
# plt.text(-0.15e+7, 4.5, '(c)', fontsize=16, fontproperties='Times New Roman')

plt.tight_layout()
plt.savefig('logic_XOR.svg', format='svg')
data = {'time': time, 'Jin1': Jin2_1, 'Jin2': Jin2_2, 'Vmem1': output_E1, 'Vmem2': output_E2, 'Jout': Jout_post}
df = pd.DataFrame(data)
df.to_excel('output_logic.xlsx', index=False)
##########################################XOR_绘图############################################