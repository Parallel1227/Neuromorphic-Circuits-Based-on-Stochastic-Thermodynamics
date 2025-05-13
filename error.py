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


def NEURON(Jin, Vout, Vth, E0, flag, St, delta_E0, current_state):
    A = zeros((2, 2))
    Gamma_l = 0.2
    Gamma_r = 0.2
    Gamma_R = 0.002

    kBT = 1.0
    Cg = 20

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

    ##############################均值###################################
    # p = null_space(A)
    # sum = p[0][0] + p[1][0]
    # p[0][0] = p[0][0] / sum
    # p[1][0] = p[1][0] / sum
    # p_N = p[1][0]
    ##############################均值###################################

    ##############################单次轨迹###################################
    # 根据当前状态计算转移速率
    if current_state == 1:
        rate = k_lN + k_rN  # 从状态1转移到状态0的速率
    else:
        rate = k_Nl + k_Nr  # 从状态0转移到状态1的速率

    # 计算转移概率
    # prob = 1 - np.exp(-rate * tint)
    prob = rate     #prob = rate * tint，tint只能为1，此处省略，提交代码时需更改
    if np.random.rand() < prob:
        current_state = 1 - current_state  # 切换状态
    # print(prob)
    # 根据当前状态计算J_d
    p_N = current_state
    ##############################单次轨迹###################################

    J_d = k_rN * p_N - k_Nr * (1 - p_N)
    J_GND = 0.5 * Gamma_R * (Fermi((mu_l - mu_l) / kBT) - Fermi((mu_l - mu_r) / kBT))

    if (flag == 0):
        Vout += 1.0 * (Jin - J_d - J_GND) * tint / Cg
    else:
        Vout += 1.0 * (0.0 - J_d - J_GND) * tint / Cg

    diss_MOS = 1.0 * J_d * tint * (mu_l - mu_r)
    diss_GND = 1.0 * J_GND * tint * (mu_l - mu_r)

    return J_d, J_GND, Vout, Vg, flag, diss_MOS, diss_GND, current_state


##########################################均值_轨迹_数据############################################
# Vout_E = 0.0
# Vout_I = 0.0
# E0 = 1.0
# delta_E0 = -0.0
#
# tint = 10
# T = 300000
# Ntot = int(T / tint)
# t = np.linspace(0, T, tint)
# Vth = 1.0
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
# Jout_post = np.zeros(Ntot)
#
# Jin = np.zeros(Ntot)
# J_GND = np.zeros(Ntot)
# start_t = 10000
# delta_t = 5000
# Jin[start_t:start_t + delta_t] = 0.05
# current_state = 0
# n_off = 0
# for i in range(Ntot):
#     output_E[i] = Vout_E
#     output_I[i] = Vout_I
#     Jout[i], J_GND[i], Vout_E, Vg_E_R, flag_E, diss_MOS[i], diss_GND[i], current_state = NEURON(Jin[i], Vout_E, Vth, E0, flag_E, 1, delta_E0, current_state)
#     Vg_E[i] = Vg_E_R
#     flag[i] = flag_E
#     diss_MOS_R = diss_MOS[i] + diss_MOS_R
#     energy_MOS[i] = diss_MOS_R
#     diss_GND_R = diss_GND[i] + diss_GND_R
#     energy_GND[i] = diss_GND_R
#     time[i] = i * tint
#     # if(flag_E == 1):
#     #     break
#     # if(i > 0):
#     #     if(output_E[i-1] >= Vth * 0.9)&(output_E[i] < Vth * 0.9):
#     #         print(i-1-start_t, output_E[i-1])
#     if(i > 0):
#         if(flag[i] == 1)&(n_off <= delta_t):
#             Jout_post[i] = 0.05
#         if(flag[i] == 1):
#             n_off = n_off + 1
#         if(flag[i] == 0):
#             n_off = 0
#
#
#
# # print(diss_MOS_R, diss_GND_R, diss_MOS_R + diss_GND_R)
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
# # plt.plot(time, Jout, color='orange')
# # plt.plot(time, J_GND, color='cornflowerblue')
# plt.plot(time, Jout_post, color='orange')
# # print(max(output_E),np.argmax(output_E),output_E[np.argmax(output_E)+1])
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
# plt.savefig('供电电压_噪声.svg', format='svg')
# data = {'time': time, 'Vmem_轨迹': output_E, 'Iout_轨迹': Jout_post}
# df = pd.DataFrame(data)
# df.to_excel('output_供电电压_噪声.xlsx', index=False)
##########################################均值_轨迹_数据############################################

##########################################均值_轨迹_数据绘图############################################
#
# file_path = "D:\信息与能量研究方向\论文\基于随机热力学的神经形态电路\神经元.xlsx"
# # 使用 usecols 选择列，skiprows 跳过不需要的行
# data = pd.read_excel(file_path, usecols='A:Q', skiprows=1, nrows=30001, sheet_name='噪声')  # 从第2行起，读取A列到C列，5行数据
# # print(data)
# plt.figure(1, figsize=(3.7, 2.4), dpi=300)
# plt.rcParams['axes.formatter.min_exponent'] = 1  # 将最小指数更改为1
# plt.rc('font', size=8, family='Times New Roman')  # 将字体大小更改为12
# plt.rcParams['xtick.direction'] = 'in'  # 将x周的刻度线方向设置向内
# plt.rcParams['ytick.direction'] = 'in'  # 将y轴的刻度方向设置向内
#
# time = data['time']
#
# plt.subplot(2, 3, 1)
# plt.plot(time * 1e-5, data['Vmem_轨迹_5V'], color='#374C6D', linewidth=1, alpha=0.3)
# plt.plot(time * 1e-5, data['Vmem_均值_5V'], color='#374C6D', alpha=1)
# plt.xticks([])
# plt.yticks([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
# plt.subplot(2, 3, 2)
# plt.plot(time * 1e-5, data['Vmem_轨迹_1V'], color='#374C6D', linewidth=1, alpha=0.3)
# plt.plot(time * 1e-5, data['Vmem_均值_1V'], color='#374C6D', alpha=1)
# plt.xticks([])
# # plt.yticks([0.0, 0.5, 1.0])
# plt.ylim([-0.3, 1.1])
# plt.subplot(2, 3, 3)
# plt.plot(time * 1e-5, data['Vmem_轨迹_0.2V'], color='#374C6D', linewidth=1, alpha=0.3)
# plt.plot(time * 1e-5, data['Vmem_均值_0.2V'], color='#374C6D', alpha=1)
# plt.xticks([])
# # plt.yticks([0.0, 0.5, 1.0])
# plt.ylim([-0.3, 0.6])
# plt.subplot(2, 3, 4)
# plt.plot(time * 1e-5, data['Iout_轨迹_5V'], color='#374C6D', linewidth=1, alpha=0.3)
# plt.plot(time * 1e-5, data['Iout_均值_5V'], color='#374C6D', alpha=1)
# plt.xticks([0, 1, 2, 3])
# plt.yticks([0, 0.01, 0.02, 0.03, 0.04, 0.05])
# plt.ylim([-0.002, 0.052])
# plt.subplot(2, 3, 5)
# plt.plot(time * 1e-5, data['Iout_轨迹_1V'], color='#374C6D', linewidth=1, alpha=0.3)
# plt.plot(time * 1e-5, data['Iout_均值_1V'], color='#374C6D', alpha=1)
# plt.xticks([0, 1, 2, 3])
# plt.yticks([0, 0.01, 0.02, 0.03, 0.04, 0.05])
# plt.ylim([-0.002, 0.052])
# plt.subplot(2, 3, 6)
# plt.plot(time * 1e-5, data['Iout_轨迹_0.2V'], color='#374C6D', linewidth=1, alpha=0.3)
# plt.plot(time * 1e-5, data['Iout_均值_0.2V'], color='#374C6D', alpha=1)
# plt.xticks([0, 1, 2, 3])
# plt.yticks([0, 0.01, 0.02, 0.03, 0.04, 0.05])
# plt.ylim([-0.002, 0.052])
#
# plt.tight_layout()
# plt.savefig('供电电压_噪声_数据绘图.svg', format='svg')

##########################################均值_轨迹_数据绘图############################################

##########################################错误率_数据############################################
Vout_E = 0.0
Vout_I = 0.0
E0 = 1.3
delta_E0 = -0.0

tint = 10
T = 50000000
Ntot = int(T / tint)
Vth = 1.3
flag_E = 0
flag_break = 0

output_E = np.zeros(Ntot)
output_I = np.zeros(Ntot)
Jout = np.zeros(Ntot)
Vg_E = np.zeros(Ntot)
time = np.zeros(Ntot)
flag = np.zeros(Ntot)
diss_MOS = np.zeros(Ntot)
energy_MOS = np.zeros(Ntot)
diss_GND = np.zeros(Ntot)
energy_GND = np.zeros(Ntot)
diss_MOS_R = 0.0
diss_GND_R = 0.0
Jout_post = np.zeros(Ntot)
Jin = np.zeros(Ntot)
J_GND = np.zeros(Ntot)

frequency = T / tint * 1e-5  # 频率 (Hz)
duty_cycle = 50  # 占空比 (%)
t = np.linspace(0, T, Ntot)
Jin = (np.mod(t * frequency, 1) < (duty_cycle / 100)) * 0.05

# start_t = 5000
delta_t = 5000
# Jin[start_t:start_t + delta_t] = 0.05
# Jin[start_t + 3*delta_t:start_t + 4*delta_t] = 0.05
Ndec = int(Ntot / delta_t)
n_spike = np.zeros(Ndec)
current_state = 0
n_off = 0
for i in range(Ntot):
    output_E[i] = Vout_E
    output_I[i] = Vout_I
    Jout[i], J_GND[i], Vout_E, Vg_E_R, flag_E, diss_MOS[i], diss_GND[i], current_state = NEURON(Jin[i], Vout_E, Vth, E0, flag_E, 1, delta_E0, current_state)
    Vg_E[i] = Vg_E_R
    flag[i] = flag_E
    diss_MOS_R = diss_MOS[i] + diss_MOS_R
    energy_MOS[i] = diss_MOS_R
    diss_GND_R = diss_GND[i] + diss_GND_R
    energy_GND[i] = diss_GND_R
    time[i] = i * tint
    j = int(i / delta_t)
    print(f"{j/10}%")
    if(i > 0):
        if(flag[i] == 1)&(n_off <= delta_t):
            Jout_post[i] = 0.05
        if(flag[i] == 1):
            n_off = n_off + 1
        if(flag[i] == 0):
            n_off = 0
        if(Jout_post[i - 1] == 0)&(Jout_post[i] != 0):
            n_spike[j] = 1
# print(Ndec, j, n_spike)


plt.figure(1, figsize=(15, 10), dpi=300)
plt.rcParams['axes.formatter.min_exponent'] = 1  # 将最小指数更改为1
plt.rc('font', size=16, family='Times New Roman')  # 将字体大小更改为12
plt.rcParams['xtick.direction'] = 'in'  # 将x周的刻度线方向设置向内
plt.rcParams['ytick.direction'] = 'in'  # 将y轴的刻度方向设置向内
plt.subplot(2, 2, 1)
plt.plot(time, Jin, color='orange')
plt.xlabel("Time")
plt.ylabel("Jin")
plt.subplot(2, 2, 2)
plt.plot(time, output_E, color='orange')
# plt.plot(time, output_I, color='cornflowerblue')
plt.xlabel("Time")
plt.ylabel("Vout")
plt.subplot(2, 2, 3)
# plt.plot(time, Jout, color='orange')
# plt.plot(time, J_GND, color='cornflowerblue')
plt.plot(time, Jout_post, color='orange')
# print(max(output_E),np.argmax(output_E),output_E[np.argmax(output_E)+1])

plt.xlabel("Time")
plt.ylabel("J")
plt.subplot(2, 2, 4)
plt.plot(time, energy_MOS, color='orange')
plt.plot(time, energy_GND, color='cornflowerblue')
plt.plot(time, energy_MOS + energy_GND, color='black')
plt.xlabel("Time")
plt.ylabel("diss")

plt.tight_layout()
plt.savefig('供电电压_噪声.svg', format='svg')
data = {'n_spike': n_spike}
df = pd.DataFrame(data)
df.to_excel('output_供电电压_噪声.xlsx', index=False)
##########################################错误率_数据############################################

##########################################增加并行神经元数量，错误率下降，但是总能耗上升############################################
# E0 = 0.4
# delta_E0 = -0.0
#
# tint = 10
# T = 500000
# Ntot = int(T / tint)
# Vth = 0.4
# flag_break = 0
#
# output_E1 = np.zeros(Ntot)
# output_E2 = np.zeros(Ntot)
# output_E3 = np.zeros(Ntot)
# output_E4 = np.zeros(Ntot)
# output_E = np.zeros(Ntot)
# Jout = np.zeros(Ntot)
# J_GND = np.zeros(Ntot)
# Vout_E1 = 0.0
# Vg_E1 = np.zeros(Ntot)
# flag_E1 = 0
# Vout_E2 = 0.0
# Vg_E2 = np.zeros(Ntot)
# flag_E2 = 0
# Vout_E3 = 0.0
# Vg_E3 = np.zeros(Ntot)
# flag_E3 = 0
# Vout_E4 = 0.0
# Vg_E4 = np.zeros(Ntot)
# flag_E4 = 0
# Vout_E = 0.0
# Vg_E = np.zeros(Ntot)
# flag_E = 0
# time = np.zeros(Ntot)
# flag1 = np.zeros(Ntot)
# flag2 = np.zeros(Ntot)
# flag3 = np.zeros(Ntot)
# flag4 = np.zeros(Ntot)
# flag = np.zeros(Ntot)
# diss_MOS = np.zeros(Ntot)
# energy_MOS = np.zeros(Ntot)
# diss_GND = np.zeros(Ntot)
# energy_GND = np.zeros(Ntot)
# diss_MOS_R = 0.0
# diss_GND_R = 0.0
# Jout_post1 = np.zeros(Ntot)
# Jout_post2 = np.zeros(Ntot)
# Jout_post3 = np.zeros(Ntot)
# Jout_post4 = np.zeros(Ntot)
# Jout_post = np.zeros(Ntot)
# Jin = np.zeros(Ntot)
#
# frequency = T / tint * 1e-5  # 频率 (Hz)
# duty_cycle = 50  # 占空比 (%)
# t = np.linspace(0, T, Ntot)
# Jin = (np.mod(t * frequency, 1) < (duty_cycle / 100)) * 0.05
#
# # start_t = 5000
# delta_t = 5000
# # Jin[start_t:start_t + delta_t] = 0.05
# # Jin[start_t + 3*delta_t:start_t + 4*delta_t] = 0.05
# Ndec = int(Ntot / delta_t)
# n_spike = np.zeros(Ndec)
# current_state1 = 0
# current_state2 = 0
# current_state3 = 0
# current_state4 = 0
# current_state = 0
# n_off1 = 0
# n_off2 = 0
# n_off3 = 0
# n_off4 = 0
# n_off = 0
# for i in range(Ntot):
#     output_E1[i] = Vout_E1
#     output_E2[i] = Vout_E2
#     output_E3[i] = Vout_E3
#     output_E4[i] = Vout_E4
#     output_E[i] = Vout_E
#     _, _, Vout_E1, Vg_E_R1, flag_E1, _, _, current_state1 = NEURON(Jin[i], Vout_E1, Vth, E0, flag_E1, 1, delta_E0, current_state1)
#     _, _, Vout_E2, Vg_E_R2, flag_E2, _, _, current_state2 = NEURON(Jin[i], Vout_E2, Vth, E0, flag_E2, 1, delta_E0, current_state2)
#     _, _, Vout_E3, Vg_E_R3, flag_E3, _, _, current_state3 = NEURON(Jin[i], Vout_E3, Vth, E0, flag_E3, 1, delta_E0, current_state3)
#     _, _, Vout_E4, Vg_E_R4, flag_E4, _, _, current_state4 = NEURON(Jin[i], Vout_E4, Vth, E0, flag_E4, 1, delta_E0, current_state4)
#     if(i > 0):
#         _, _, Vout_E, Vg_E_R, flag_E, _, _, current_state = NEURON(Jout_post1[i - 1] + Jout_post2[i - 1] + Jout_post3[i - 1] + Jout_post4[i - 1], Vout_E, Vth, E0, flag_E, 1, delta_E0, current_state)
#         Vg_E[i] = Vg_E_R
#         flag[i] = flag_E
#         # print(Jout_post1[i - 1] + Jout_post2[i - 1])
#     Vg_E1[i] = Vg_E_R1
#     flag1[i] = flag_E1
#     Vg_E2[i] = Vg_E_R2
#     flag2[i] = flag_E2
#     Vg_E3[i] = Vg_E_R3
#     flag3[i] = flag_E3
#     Vg_E4[i] = Vg_E_R4
#     flag4[i] = flag_E4
#     # diss_MOS_R = diss_MOS[i] + diss_MOS_R
#     # energy_MOS[i] = diss_MOS_R
#     # diss_GND_R = diss_GND[i] + diss_GND_R
#     # energy_GND[i] = diss_GND_R
#     time[i] = i * tint
#     j = int(i / delta_t)
#     print(f"{j/10}%")
#
#     if(flag1[i] == 1)&(n_off1 <= delta_t):
#         Jout_post1[i] = 0.0125
#     if(flag1[i] == 1):
#         n_off1 = n_off1 + 1
#     if(flag1[i] == 0)&(n_off1 >= delta_t):
#         n_off1 = 0
#     if (flag2[i] == 1) & (n_off2 <= delta_t):
#         Jout_post2[i] = 0.0125
#     if (flag2[i] == 1):
#         n_off2 = n_off2 + 1
#     if (flag2[i] == 0) & (n_off2 >= delta_t):
#         n_off2 = 0
#     if (flag3[i] == 1) & (n_off3 <= delta_t):
#         Jout_post3[i] = 0.0125
#     if (flag3[i] == 1):
#         n_off3 = n_off3 + 1
#     if (flag3[i] == 0) & (n_off3 >= delta_t):
#         n_off3 = 0
#     if (flag4[i] == 1) & (n_off4 <= delta_t):
#         Jout_post4[i] = 0.0125
#     if (flag4[i] == 1):
#         n_off4 = n_off4 + 1
#     if (flag4[i] == 0) & (n_off4 >= delta_t):
#         n_off4 = 0
#     if (flag[i] == 1) & (n_off <= delta_t):
#         Jout_post[i] = 0.05
#     if (flag[i] == 1):
#         n_off = n_off + 1
#     if (flag[i] == 0) & (n_off >= delta_t):
#         n_off = 0
#     if (i > 0):
#         if(Jout_post[i - 1] == 0)&(Jout_post[i] != 0):
#             n_spike[j] = 1
# # print(Ndec, j, n_spike)
#
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
# plt.plot(time, output_E1, color='orange')
# plt.plot(time, output_E2, color='cornflowerblue')
# plt.plot(time, output_E, color='black')
# plt.xlabel("Time")
# plt.ylabel("Vout")
# plt.subplot(2, 2, 3)
# # plt.plot(time, Jout, color='orange')
# # plt.plot(time, J_GND, color='cornflowerblue')
# plt.plot(time, Jout_post1, color='orange')
# plt.plot(time, Jout_post2, color='cornflowerblue')
# plt.plot(time, Jout_post, color='black')
# # print(max(output_E),np.argmax(output_E),output_E[np.argmax(output_E)+1])
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
# plt.savefig('供电电压_噪声.svg', format='svg')
# data = {'n_spike': n_spike}
# df = pd.DataFrame(data)
# df.to_excel('output_供电电压_噪声.xlsx', index=False)
##########################################增加并行神经元数量，错误率下降，但是总能耗上升############################################