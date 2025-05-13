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


# def NEURON(Jin, Vout, Vth, E0, flag, St, delta_E0):
#     A = zeros((2, 2))
#     Gamma_l = 0.2
#     Gamma_r = 0.2
#     Gamma_R = 0.002
#
#     kBT = 1.0
#     Cg = 200.0
#
#     # if(St == 0):
#     # 	Vg = 5.0
#     # else:
#     # 	if (Vout < Vth):
#     # 		Vg = 0.0
#     # 	else:
#     # 		Vg = 5.0
#
#     if (flag == 0):
#         if (Vout < Vth):
#             Vg = 0.0
#         else:
#             Vg = E0
#             flag = 1
#     else:
#         if (Vout <= (0.001)):
#             Vg = 0.0
#             flag = 0
#         else:
#             Vg = E0
#
#     if (St == 0):
#         Vg = E0
#
#     # print(flag, Vg, Vout)
#
#     E_N = E0 - Vg + delta_E0
#
#     mu_l = 0.0
#     mu_r = mu_l - Vout
#
#     k_Nl = Gamma_l * Fermi((E_N - mu_l) / kBT)
#     k_lN = Gamma_l * (1.0 - Fermi((E_N - mu_l) / kBT))
#     k_rN = Gamma_r * (1.0 - Fermi((E_N - mu_r) / kBT))
#     k_Nr = Gamma_r * Fermi((E_N - mu_r) / kBT)
#
#     A[1][0] = k_Nr + k_Nl
#     A[0][0] = -A[1][0]
#     A[0][1] = k_rN + k_lN
#     A[1][1] = -A[0][1]
#
#     p = null_space(A)
#     sum = p[0][0] + p[1][0]
#     p[0][0] = p[0][0] / sum
#     p[1][0] = p[1][0] / sum
#     p_N = p[1][0]
#     J_d = k_rN * p_N - k_Nr * (1 - p_N)
#     J_GND = 0.5 * Gamma_R * (Fermi((mu_l - mu_l) / kBT) - Fermi((mu_l - mu_r) / kBT))
#     # 电压控制的开关 放电时屏蔽输入电流
#     if (flag == 0):
#         Vout += 1.0 * (Jin - J_d - J_GND) * tint / Cg
#     else:
#         Vout += 1.0 * (0.0 - J_d - J_GND) * tint / Cg
#         # diss_MOS = 1.0 * J_d * tint * (mu_l - mu_r)
#         # diss_GND = J_GND * J_GND * 8.432 / Gamma_R
#     diss_MOS = 1.0 * J_d * tint * (mu_l - mu_r)
#     diss_GND = 1.0 * J_GND * tint * (mu_l - mu_r)
#
#     return Vout, Vg, flag, diss_MOS + diss_GND


##########################################test############################################
#Vout_E = 0.0
# Vout_I = 0.0
# E0 = 1.0
#
# tint = 10
# T = 1000000
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
# start_t = 20000
# delta_t = 2037
# Jin[start_t:start_t + delta_t] = 0.1
# # Jin[20000:21000] = 0.05
# # Jin[30000:31000] = 0.05
# # Jin[40000:41000] = 0.05
#
# for i in range(Ntot):
#     output_E[i] = Vout_E
#     output_I[i] = Vout_I
#     Jout[i], J_GND[i], Vout_E, Vg_E_R, flag_E, diss_MOS[i], diss_GND[i] = NEURON(Jin[i], Vout_E, Vth, E0, flag_E, 1, 0.0)
#     Vg_E[i] = Vg_E_R
#     flag[i] = flag_E
#     diss_MOS_R = diss_MOS[i] + diss_MOS_R
#     energy_MOS[i] = diss_MOS_R
#     diss_GND_R = diss_GND[i] + diss_GND_R
#     energy_GND[i] = diss_GND_R
#     time[i] = i * tint
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
# plt.savefig('logic.svg', format='svg')

##########################################test############################################

##########################################能耗数据############################################
# tint = 10
# T = 1000000
# Ntot = int(T / tint)
# t = np.linspace(0, T, tint)
# Vth = 1.0
# E0 = 1.0
# diss_sum = 0.0
# time = np.zeros(Ntot)
#
# Vout_dog = 0.0
# output_dog = np.zeros(Ntot)
# Vg_dog = np.zeros(Ntot)
# Vg_dog_R = 0
# flag_dog = np.zeros(Ntot)
# flag_dog_R = 0
# diss_dog = np.zeros(Ntot)
# Vout_brown = 0.0
# output_brown = np.zeros(Ntot)
# Vg_brown = np.zeros(Ntot)
# Vg_brown_R = 0
# flag_brown = np.zeros(Ntot)
# flag_brown_R = 0
# diss_brown = np.zeros(Ntot)
# Vout_7 = 0.0
# output_7 = np.zeros(Ntot)
# Vg_7 = np.zeros(Ntot)
# Vg_7_R = 0
# flag_7 = np.zeros(Ntot)
# flag_7_R = 0
# diss_7 = np.zeros(Ntot)
#
# t_dog_7 = 219
# t_brown_7 = 219
# t_7 = 486
# J_dog_7 = np.zeros(Ntot)
# J_brown_7 = np.zeros(Ntot)
#
# Jin_brown = np.zeros(Ntot)
# Jin_dog = np.zeros(Ntot)
# Jin_7 = np.zeros(Ntot)
# Jout_7 = np.zeros(Ntot)
# Jin_brown[20000:20486] = 0.05
# Jin_dog[20000:20486] = 0.05
#
# for i in range(Ntot):
#     # neuron_dog
#     output_dog[i] = Vout_dog
#     Vout_dog, Vg_dog_R, flag_dog_R, diss_dog[i] = NEURON(Jin_dog[i], Vout_dog, Vth, E0, flag_dog_R, 1, 0.0)
#     Vg_dog[i] = Vg_dog_R
#     flag_dog[i] = flag_dog_R
#     diss_sum = diss_dog[i] + diss_sum
#     if (i > 0):
#         if (flag_dog[i - 1] == 0) & (flag_dog[i] == 1):
#             J_dog_7[i:i + t_dog_7] = 0.05
#
#     # neuron_brown
#     output_brown[i] = Vout_brown
#     Vout_brown, Vg_brown_R, flag_brown_R, diss_brown[i] = NEURON(Jin_brown[i], Vout_brown, Vth, E0, flag_brown_R, 1, 0.0)
#     Vg_brown[i] = Vg_brown_R
#     flag_brown[i] = flag_brown_R
#     diss_sum = diss_brown[i] + diss_sum
#     if (i > 0):
#         if (flag_brown[i - 1] == 0) & (flag_brown[i] == 1):
#             J_brown_7[i:i + t_brown_7] = 0.05
#
#     # neuron_7
#     output_7[i] = Vout_7
#     Jin_7[i] = J_dog_7[i] + J_brown_7[i]
#     Vout_7, Vg_7_R, flag_7_R, diss_7[i] = NEURON(Jin_7[i], Vout_7, Vth, E0, flag_7_R, 1, 0.0)
#     Vg_7[i] = Vg_7_R
#     flag_7[i] = flag_7_R
#     diss_sum = diss_7[i] + diss_sum
#     if (i > 0):
#         if (flag_7[i - 1] == 0) & (flag_7[i] == 1):
#             Jout_7[i:i + t_7] = 0.05
#             print(i * tint)
#         if (flag_7[i - 1] == 1) & (flag_7[i] == 0):
#             print(i * tint)
#
#     time[i] = i * tint
#
# print(diss_sum)
#
# plt.figure(1, figsize=(10, 8), dpi=300)
# plt.rcParams['axes.formatter.min_exponent'] = 1  # 将最小指数更改为1
# plt.rc('font', size=14, family='Times New Roman')  # 将字体大小更改为16
# plt.rcParams['xtick.direction'] = 'in'  # 将x周的刻度线方向设置向内
# plt.rcParams['ytick.direction'] = 'in'  # 将y轴的刻度方向设置向内
#
# subplot(4, 1, 1)
# plot(time, Jin_brown, color='#4c6792', linewidth='2')
# plot(time, Jin_dog, linestyle='--', color='red', linewidth='2')
# plt.xlabel("Time ($β\hbar$)")
# plt.ylabel("Jin")
# # plt.text(-0.15e+7, 5, '(a)', fontsize=16, fontproperties='Times New Roman')
#
# subplot(4, 1, 2)
# plot(time, J_dog_7, color='#4c6792', linewidth='2')
# plot(time, J_brown_7, linestyle='--', color='red', linewidth='2')
# plt.xlabel("Time ($β\hbar$)")
# plt.ylabel("J_neuron_7")
# # plt.text(-0.15e+7, 5, '(b)', fontsize=16, fontproperties='Times New Roman')
#
# subplot(4, 1, 3)
# plot(time, output_dog, color='#4c6792', linewidth='2')
# plot(time, output_brown, linestyle='--', color='red', linewidth='2')
# plot(time, output_7, linestyle='--', color='black', linewidth='2')
# plt.xlabel("Time ($β\hbar$)")
# plt.ylabel("Vout")
# # plt.text(-0.15e+7, 4.5, '(c)', fontsize=16, fontproperties='Times New Roman')
#
# subplot(4, 1, 4)
# plot(time, Jout_7, color='#4c6792', linewidth='2')
# plt.xlabel("Time ($β\hbar$)")
# plt.ylabel("Jout_post")
# # plt.text(-0.15e+7, 4.5, '(c)', fontsize=16, fontproperties='Times New Roman')
#
#
# plt.tight_layout()
# plt.savefig('test.svg', format='svg')
# # data = {'time': time, 'Jin1': Jin1, 'Jin2': Jin2, 'Vmem': output_E, 'Jout': Jout_post}
# # df = pd.DataFrame(data)
# # df.to_excel('output_logic.xlsx', index=False)
##########################################能耗数据############################################

##########################################对比绘图############################################

input_bit = ('CMOS','NC')

W_CMOS = 1940742
W_NC = 371

# plt.rc('font',family = 'Times New Roman')
# plt.figure(1, dpi=300)
# plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签

x0 = np.arange(-0.125, 0.625, 0.001)
y0 = W_CMOS * np.ones(750)
x1 = np.arange(0.375, 0.625, 0.001)
y1 = W_NC * np.ones(250)


plt.figure(1, figsize=(3, 2.9), dpi=300)

plt.rcParams['xtick.direction'] = 'in'#将x周的刻度线方向设置向内
plt.rcParams['ytick.direction'] = 'in'#将y轴的刻度方向设置向内
plt.subplot(1, 2, 1)
bar_width = 0.25  # 条形宽度
index_max = np.arange(float(len(input_bit)))  # max条形图的横坐标
# plt.grid(visible="True",axis="y",zorder=0)
index_max[1] -= 0.5
# index_max[2] -= 1.0
index_min = index_max + bar_width  # min条形图的横坐标
plt.bar(index_max[0], height=W_CMOS, width=bar_width, color='#374C6D', zorder=100)
plt.bar(index_max[1], height=W_NC, width=bar_width, color='#a3abbd', zorder=100)
plt.xticks(index_max, input_bit, fontsize=9, fontproperties='Times New Roman')
plt.yticks(fontsize=8, fontproperties='Times New Roman')
# plt.ylabel("Energy consumptions of XOR gate for one operation($×10^{-17} Joule$)",fontsize=13,fontproperties='Times New Roman')

plt.plot(x0, y0, color='orange', linewidth=1.5, zorder=100)
plt.plot(x1, y1, color='orange', linewidth=1.5, zorder=100)

plt.annotate('', xy=(0.5, 336), xytext=(0.5, 2150742), arrowprops=dict(arrowstyle='<->', color="orange"))
# plt.text(0.01, 2.18, '9.61%', color='r',fontsize=15, fontproperties='Times New Roman')
# plt.annotate('', xy=(0.5, 2.27), xytext=(0.5, 2.34), arrowprops=dict(arrowstyle='-', color="r"))
# plt.annotate('', xy=(0.501, 2.32), xytext=(0.73, 1.8), arrowprops=dict(arrowstyle='<-', color="r"),zorder = 101)
# plt.text(0.65, 1.73, '2.15%', color='r',fontsize=15, fontproperties='Times New Roman')
plt.xlim(-0.25, 0.75)
# plt.tick_params(labelsize=13)
plt.yscale('log')
plt.tight_layout()
plt.savefig('注意力_对比.jpg', format='jpg', bbox_inches='tight')
##########################################对比绘图############################################
