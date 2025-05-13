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
Vout_E = 0.0
Vout_I = 0.0
E0 = 1.0
delta_E0 = -0.0

tint = 10
T = 1000000
Ntot = int(T / tint)
t = np.linspace(0, T, tint)
Vth = 1.0
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

Jin = np.zeros(Ntot)
J_GND = np.zeros(Ntot)
start_t = 20000
delta_t = 500
Jin[start_t:start_t + delta_t] = 0.1
for i in range(Ntot):
    output_E[i] = Vout_E
    output_I[i] = Vout_I
    Jout[i], J_GND[i], Vout_E, Vg_E_R, flag_E, diss_MOS[i], diss_GND[i] = NEURON(Jin[i], Vout_E, Vth, E0, flag_E, 1, delta_E0)
    Vg_E[i] = Vg_E_R
    flag[i] = flag_E
    diss_MOS_R = diss_MOS[i] + diss_MOS_R
    energy_MOS[i] = diss_MOS_R
    diss_GND_R = diss_GND[i] + diss_GND_R
    energy_GND[i] = diss_GND_R
    time[i] = i * tint
    if(i > 0):
        if(output_E[i-1] >= Vth * 0.9)&(output_E[i] < Vth * 0.9):
            print(i-1-start_t, output_E[i-1])

print(diss_MOS_R, diss_GND_R, diss_MOS_R + diss_GND_R)

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
plt.plot(time, Jout, color='orange')
plt.plot(time, J_GND, color='cornflowerblue')
print(max(output_E),np.argmax(output_E),output_E[np.argmax(output_E)+1])

plt.xlabel("Time")
plt.ylabel("J")
plt.subplot(2, 2, 4)
plt.plot(time, energy_MOS, color='orange')
plt.plot(time, energy_GND, color='cornflowerblue')
plt.plot(time, energy_MOS + energy_GND, color='black')
plt.xlabel("Time")
plt.ylabel("diss")

plt.tight_layout()
plt.savefig('test.svg', format='svg')

##########################################test############################################

##########################################兴奋/抑制_调控延时_波形############################################
# from matplotlib.colors import LinearSegmentedColormap
#
#
# Vout_E = np.zeros(10)
# E0 = 5.0
# delta_E0 = -4.5
# Vctrl = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]
#
# tint = 10
# T = 100000
# Ntot = int(T / tint)
# t = np.linspace(0, T, tint)
# Vth = 5.0
# flag_E = np.zeros(10)
# flag_break = 0
#
# output_E = np.zeros((10, Ntot))
# # Jout = np.zeros(Ntot)
# # Vg_E = np.zeros(Ntot)
# time = np.zeros(Ntot)
# flag = np.zeros((10, Ntot))
# # diss_MOS = np.zeros(Ntot)
# # energy_MOS = np.zeros(Ntot)
# # diss_GND = np.zeros(Ntot)
# # energy_GND = np.zeros(Ntot)
# # diss_MOS_R = 0.0
# # diss_GND_R = 0.0
#
# Jin = np.zeros(Ntot)
# J_GND = np.zeros(Ntot)
# start_t = 0
# delta_t = 5636
# Jin[start_t:start_t + delta_t] = 0.05
#
# for j in range(10):
#     for i in range(Ntot):
#         output_E[j][i] = Vout_E[j]
#         _, _, Vout_E[j], _, flag_E[j], _, _ = NEURON(Jin[i], Vout_E[j], Vth, E0, flag_E[j], 1, -Vctrl[j])
#         flag[j][i] = flag_E[j]
#         # Vg_E[i] = Vg_E_R
#         # diss_MOS_R = diss_MOS[i] + diss_MOS_R
#         # energy_MOS[i] = diss_MOS_R
#         # diss_GND_R = diss_GND[i] + diss_GND_R
#         # energy_GND[i] = diss_GND_R
#         time[i] = i * tint
#         # if(i > 0):
#         #     if(output_E[i-1] >= Vth * 0.9)&(output_E[i] < Vth * 0.9):
#         #         print(i-1-start_t, output_E[i-1])
#
# # print(diss_MOS_R, diss_GND_R, diss_MOS_R + diss_GND_R)
#
# # plt.figure(1, figsize=(15, 10), dpi=300)
# plt.rcParams['axes.formatter.min_exponent'] = 1  # 将最小指数更改为1
# plt.rc('font', size=8, family='Times New Roman')  # 将字体大小更改为12
# plt.rcParams['xtick.direction'] = 'in'  # 将x周的刻度线方向设置向内
# plt.rcParams['ytick.direction'] = 'in'  # 将y轴的刻度方向设置向内
#
#
# data = output_E
#
# # 自定义10组数据的名称
# group_labels = [f"{i * 0.5:.1f}" for i in range(10)]
#
# # 颜色映射（colormap）
# cmap = get_cmap("Blues")  # 可选 'viridis', 'plasma', 'coolwarm', 等等
# colors = cmap(np.linspace(0, 1, len(data)))
#
# # # 创建自定义的渐变颜色映射
# # colors_hex = ["#5CA6FF", "#004496"]
# # custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", colors_hex)
# #
# # # 为每一组数据生成对应的颜色
# # colors = custom_cmap(np.linspace(0, 1, len(data)))
#
# # 绘制山峦图
# fig, ax = plt.subplots(figsize=(2.2, 2.2), dpi=300)
#
# # # 添加网格线
# # ax.grid(axis="y", linestyle="-", alpha=0.7, linewidth=1)
#
# # 反转数据和颜色顺序，确保0.0在最顶层
# for i, (voltages, color) in reversed(list(enumerate(zip(data, colors)))):
#     y_base = i * 1.5  # 每组的基线高度
#     ax.plot(time, voltages + y_base, color=color, alpha=0.8)  # 绘制曲线
#     ax.fill_between(time, voltages + y_base, y_base, color=color, alpha=0.6)  # 填充曲线下方
#
# # 设置y轴刻度和标签
# ax.set_yticks([i * 1.5 for i in range(len(data))])
# ax.set_yticklabels(group_labels)
#
# plt.xlim(0, 70000)
# plt.ylim(0, 19)
#
# # 设置标题和坐标轴标签
# ax.set_xlabel("Time ($β\hbar$)", fontsize=9, fontproperties='Times New Roman')
# ax.set_ylabel("$V_{\mathrm{ctrl}}$ ($V_{T}$)", fontsize=9, fontproperties='Times New Roman')
# # ax.set_title("Ridge Plot: Voltage Over Time with Colormap (Reversed Layers)")
#
# # 美化图表布局
# plt.tight_layout()
# # plt.savefig('兴奋抑制_延时_波形.svg', format='svg')
# plt.show()
##########################################兴奋/抑制_调控延时_波形############################################

##########################################兴奋/抑制_调控延时_数据############################################
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

##########################################兴奋/抑制_调控延时_数据############################################

##########################################兴奋/抑制_调控延时_数据绘图############################################
#
# file_path = "D:\信息与能量研究方向\论文\基于随机热力学的神经形态电路\神经元.xlsx"
# # 使用 usecols 选择列，skiprows 跳过不需要的行
# data = pd.read_excel(file_path, usecols='A, C:D', skiprows=37, nrows=11, sheet_name='兴奋抑制')  # 从第2行起，读取A列到C列，5行数据
#
# Vctrl = -data['delta_E0']
# delta_t = data['delta']
# diss = data['energy_sum']
#
# # # 数据图1
# # plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
# # plt.figure(1, dpi=300, figsize=(1.8, 2.1))
# # plt.rcParams['xtick.direction'] = 'in'  # 将x周的刻度线方向设置向内
# # plt.rcParams['ytick.direction'] = 'in'  # 将y轴的刻度方向设置向内
# # plt.subplot(2, 1, 1)
# # plt.plot(Vctrl, delta_t, label='$\Delta t_{\mathrm{dev}}$($β\hbar$)', marker="s", markersize=4, color='#374C6D')
# # plt.yticks(fontproperties='Times New Roman')
# # plt.xticks([])
# # plt.tick_params(labelsize=8)
# # # plt.gca().invert_xaxis()
# # plt.legend(loc='upper left', fontsize=9)
# # plt.subplot(2, 1, 2)
# # plt.plot(Vctrl, diss, label='$E_{\mathrm{neuron}}$($kT$)', marker="s", markersize=4, color='#a3abbd')
# # plt.xlabel("$V_{\mathrm{ctrl}}$($V_{T}$)", fontsize=9, fontproperties='Times New Roman')
# # plt.yticks(fontproperties='Times New Roman')
# # plt.xticks(fontproperties='Times New Roman')
# # plt.tick_params(labelsize=8)
# #
# # plt.legend(loc='upper left', fontsize=8)
# # # plt.tight_layout()
# # plt.savefig('兴奋抑制_延时_数据绘图1.svg', format='svg', bbox_inches='tight')
#
# # 数据图2
# plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
# plt.rc('font', size=8,family='Times New Roman')
# plt.figure(1, dpi=300, figsize=(1.8, 2.1))
# plt.rcParams['xtick.direction'] = 'in'  # 将x周的刻度线方向设置向内
# plt.rcParams['ytick.direction'] = 'in'  # 将y轴的刻度方向设置向内
#
# plt.scatter(delta_t, diss, label='Data points',s = 10,marker='s',c='#374C6D',edgecolors='#374C6D',zorder=10)
# plt.plot(delta_t, 1.663 * delta_t + 2536, label='Fitting result', color='#a3abbd')
# plt.xlabel("$\Delta t_{\mathrm{dev}}$($β\hbar$)", fontsize=9, fontproperties='Times New Roman')
# plt.ylabel("$E_{\mathrm{neuron}}$($kT$)", fontsize=9, fontproperties='Times New Roman')
# plt.yticks(fontproperties='Times New Roman')
# plt.xticks(fontproperties='Times New Roman')
# plt.tick_params(labelsize=8)
# plt.legend(loc='lower right', fontsize=8)
# # plt.tight_layout()
# # plt.savefig('兴奋抑制_延时_数据绘图2.svg', format='svg', bbox_inches='tight')
# plt.show()
##########################################兴奋/抑制_调控延时_数据绘图############################################

##########################################兴奋/抑制_权重控制_数据############################################
# Vout_E1 = 0.0
# Vout_E2 = 0.0
# Vout_E3 = 0.0
# E0 = 5.0
# # delta_E0 = -1.0
#
# tint = 10
# T = 1000000
# Ntot = int(T / tint)
# t = np.linspace(0, T, tint)
# Vth = 5.0
# flag_E1 = 0
# flag_E2 = 0
# flag_E3 = 0
# flag_break = 0
#
# output_E1 = np.zeros(Ntot)
# output_E2 = np.zeros(Ntot)
# output_E3 = np.zeros(Ntot)
# Jout2 = np.zeros(Ntot)
# Jout3 = np.zeros(Ntot)
# Vg_E = np.zeros(Ntot)
# time = np.zeros(Ntot)
# flag = np.zeros(Ntot)
# diss_MOS2 = np.zeros(Ntot)
# energy_MOS2 = np.zeros(Ntot)
# diss_GND2 = np.zeros(Ntot)
# energy_GND2 = np.zeros(Ntot)
# diss_MOS_R2 = 0.0
# diss_GND_R2 = 0.0
# diss_MOS3 = np.zeros(Ntot)
# energy_MOS3 = np.zeros(Ntot)
# diss_GND3 = np.zeros(Ntot)
# energy_GND3 = np.zeros(Ntot)
# diss_MOS_R3 = 0.0
# diss_GND_R3 = 0.0
# Jout_post = np.zeros(Ntot)
#
# Jin1 = np.zeros(Ntot)
# Jin2 = np.zeros(Ntot)
# Jin3 = np.zeros(Ntot)
# J_GND2 = np.zeros(Ntot)
# J_GND3 = np.zeros(Ntot)
# start_t = 30000
# delta_t = 3000
# Jin1[start_t + 4000:start_t + 7000] = 0.05    #控制Vctrl
# Jin2[start_t:start_t + 3000] = 0.05    #阈值Vth
# # Jin3[start_t + 5000:start_t + 8000] = 0.05    #输入后置Vfire
# for i in range(Ntot):
#     output_E1[i] = Vout_E1
#     output_E2[i] = Vout_E2
#     # output_E3[i] = Vout_E3
#     _, _, Vout_E1, _, flag_E1, _, _ = NEURON(Jin1[i], Vout_E1, Vth, E0, flag_E1, 1, 0.0)
#     Jout2[i], J_GND2[i], Vout_E2, Vg_E_R2, flag_E2, diss_MOS2[i], diss_GND2[i] = NEURON(Jin2[i], Vout_E2, Vth, E0, flag_E2, 1, 0.0)
#     # Jout3[i], J_GND3[i], Vout_E3, Vg_E_R3, flag_E3, diss_MOS3[i], diss_GND3[i] = NEURON(Jin3[i], Vout_E3, Vth, E0, flag_E3, 1, 0.0)
#     # Vg_E[i] = Vg_E_R
#     flag[i] = flag_E2
#     # diss_MOS_R2 = diss_MOS2[i] + diss_MOS_R2
#     # energy_MOS2[i] = diss_MOS_R2
#     # diss_GND_R2 = diss_GND2[i] + diss_GND_R2
#     # energy_GND2[i] = diss_GND_R2
#     time[i] = i * tint
#     if(i > 0):
#         if(flag[i] == 1)&(output_E2[i] >= output_E1[i]):
#             Jout_post[i] = 0.05
#
# print(diss_MOS_R2, diss_GND_R2, diss_MOS_R2 + diss_GND_R2)
#
# plt.figure(1, figsize=(15, 10), dpi=300)
# plt.rcParams['axes.formatter.min_exponent'] = 1  # 将最小指数更改为1
# plt.rc('font', size=16, family='Times New Roman')  # 将字体大小更改为12
# plt.rcParams['xtick.direction'] = 'in'  # 将x周的刻度线方向设置向内
# plt.rcParams['ytick.direction'] = 'in'  # 将y轴的刻度方向设置向内
# plt.subplot(2, 2, 1)
# plt.plot(time, Jin1, color='cornflowerblue')
# plt.plot(time, Jin2, color='orange')
# # plt.plot(time, Jin3, color='black')
# plt.xlabel("Time")
# plt.ylabel("Jin")
# plt.subplot(2, 2, 2)
# plt.plot(time, output_E1, color='cornflowerblue')
# plt.plot(time, output_E2, color='orange')
# # plt.plot(time, output_E3, color='black')
# plt.xlabel("Time")
# plt.ylabel("Vout")
# plt.xlim(0.18e+6,0.38e+6)
# plt.subplot(2, 2, 3)
# plt.plot(time, Jout_post, color='orange')
# # plt.plot(time, J_GND, color='cornflowerblue')
# # print(max(Jout))
# plt.xlabel("Time")
# plt.ylabel("Jout_post")
# plt.subplot(2, 2, 4)
# plt.plot(time, energy_MOS2, color='orange')
# plt.plot(time, energy_GND2, color='cornflowerblue')
# # plt.plot(time, energy_MOS2 + energy_GND2, color='black')
# plt.xlabel("Time")
# plt.ylabel("diss")
#
# plt.tight_layout()
# plt.savefig('兴奋抑制_权重.svg', format='svg')
# data = {'time': time[18000:38000], 'Vout_ctrl': output_E1[18000:38000], 'Vout': output_E2[18000:38000]}
# df = pd.DataFrame(data)
# df.to_excel('output_权重控制.xlsx', index=False)
##########################################兴奋/抑制_权重控制_数据############################################

##########################################兴奋/抑制_权重控制_数据绘图1############################################
# from mpl_toolkits.mplot3d.art3d import Poly3DCollection
# from matplotlib.patches import Patch
#
# file_path = "D:\信息与能量研究方向\论文\基于随机热力学的神经形态电路\神经元.xlsx"
# # 使用 usecols 选择列，skiprows 跳过不需要的行
# data = pd.read_excel(file_path, usecols='K:S', skiprows=76, nrows=20000, sheet_name='兴奋抑制')  # 从第2行起，读取A列到C列，5行数据
# # print(data)
# # 创建数据
# x = data['time4']  # x 轴数据
# z1_1 = data['Vout_ctrl4']
# z1_2 = data['Vout4']
# z2_1 = data['Vout_ctrl5']
# z2_2 = data['Vout5']
# z3_1 = data['Vout_ctrl6']
# z3_2 = data['Vout6']
# z3 = np.linspace(5, 5, 100)
#
# # 定义多个 y 值（分层）和对应的标签
# y_values = [0, 1, 2]  # 每个平面在 y 轴上的位置
# y_labels = ['100', '500', '1000']  # 自定义标签
#
# # 创建三维图形
# fig = plt.figure(figsize=(3, 2))
# ax = fig.add_subplot(111, projection='3d')
# # 绘制每个平面
# # for y, label in zip(y_values, y_labels):
# # 绘制第一条曲线
# ax.plot(x, [y_values[0]] * len(x), z1_1, color='#004496', alpha=0.2)  # 曲线
# # 绘制第二条曲线
# ax.plot(x, [y_values[0]] * len(x), z1_2, color='#374C6D', alpha=0.2)  # 曲线
# # 填充第一条曲线以下的区域
# verts1 = list(zip(x, [y_values[0]] * len(x), z1_1))  # 创建顶点
# verts1.append((x[19999], y_values[0], 0))  # 添加底部点
# verts1.append((x[0], y_values[0], 0))   # 添加底部点
# poly1 = Poly3DCollection([verts1], facecolors='#004496', alpha=0.2)  # 创建三维多边形
# ax.add_collection3d(poly1)  # 添加到三维图中
# # 填充第二条曲线以下的区域
# verts2 = list(zip(x, [y_values[0]] * len(x), z1_2))  # 创建顶点
# verts2.append((x[19999], y_values[0], 0))  # 添加底部点
# verts2.append((x[0], y_values[0], 0))   # 添加底部点
# poly2 = Poly3DCollection([verts2], facecolors='#374C6D', alpha=0.2)  # 创建三维多边形
# ax.add_collection3d(poly2)  # 添加到三维图中
# # 填充尖峰区间
# verts3 = list(zip(x[4038:5051], [y_values[0]] * len(x[4038:5051]), z3))  # 创建顶点
# verts3.append((x[4038], y_values[0], 0))  # 添加底部点
# verts3.append((x[5051], y_values[0], 0))   # 添加底部点
# verts3.append((x[5051], y_values[0], 5))   # 添加底部点
# poly3 = Poly3DCollection([verts3], facecolors='r', alpha=0.4)  # 创建三维多边形
# ax.add_collection3d(poly3)  # 添加到三维图中
#
# # 绘制第一条曲线
# ax.plot(x, [y_values[1]] * len(x), z2_1, color='#004496', alpha=0.2)  # 曲线
# # 绘制第二条曲线
# ax.plot(x, [y_values[1]] * len(x), z2_2, color='#374C6D', alpha=0.2)  # 曲线
# # 填充第一条曲线以下的区域
# verts1 = list(zip(x, [y_values[1]] * len(x), z2_1))  # 创建顶点
# verts1.append((x[19999], y_values[1], 0))  # 添加底部点
# verts1.append((x[0], y_values[1], 0))   # 添加底部点
# poly1 = Poly3DCollection([verts1], facecolors='#004496', alpha=0.2)  # 创建三维多边形
# ax.add_collection3d(poly1)  # 添加到三维图中
# # 填充第二条曲线以下的区域
# verts2 = list(zip(x, [y_values[1]] * len(x), z2_2))  # 创建顶点
# verts2.append((x[19999], y_values[1], 0))  # 添加底部点
# verts2.append((x[0], y_values[1], 0))   # 添加底部点
# poly2 = Poly3DCollection([verts2], facecolors='#374C6D', alpha=0.2)  # 创建三维多边形
# ax.add_collection3d(poly2)  # 添加到三维图中
# # 填充尖峰区间
# verts3 = list(zip(x[9038:10612], [y_values[1]] * len(x[9038:10612]), z3))  # 创建顶点
# verts3.append((x[9038], y_values[1], 0))  # 添加底部点
# verts3.append((x[10612], y_values[1], 0))   # 添加底部点
# verts3.append((x[10612], y_values[1], 5))   # 添加底部点
# poly3 = Poly3DCollection([verts3], facecolors='r', alpha=0.4)  # 创建三维多边形
# ax.add_collection3d(poly3)  # 添加到三维图中
#
# # 绘制第一条曲线
# ax.plot(x, [y_values[2]] * len(x), z3_1, color='#004496', alpha=0.2)  # 曲线
# # 绘制第二条曲线
# ax.plot(x, [y_values[2]] * len(x), z3_2, color='#374C6D', alpha=0.2)  # 曲线
# # 填充第一条曲线以下的区域
# verts1 = list(zip(x, [y_values[2]] * len(x), z3_1))  # 创建顶点
# verts1.append((x[19999], y_values[2], 0))  # 添加底部点
# verts1.append((x[0], y_values[2], 0))   # 添加底部点
# poly1 = Poly3DCollection([verts1], facecolors='#004496', alpha=0.2)  # 创建三维多边形
# ax.add_collection3d(poly1)  # 添加到三维图中
# # 填充第二条曲线以下的区域
# verts2 = list(zip(x, [y_values[2]] * len(x), z3_2))  # 创建顶点
# verts2.append((x[19999], y_values[2], 0))  # 添加底部点
# verts2.append((x[0], y_values[2], 0))   # 添加底部点
# poly2 = Poly3DCollection([verts2], facecolors='#374C6D', alpha=0.2)  # 创建三维多边形
# ax.add_collection3d(poly2)  # 添加到三维图中
# # 填充尖峰区间
# verts3 = list(zip(x[14038:16282], [y_values[2]] * len(x[14038:16282]), z3))  # 创建顶点
# verts3.append((x[14038], y_values[2], 0))  # 添加底部点
# verts3.append((x[16282], y_values[2], 0))   # 添加底部点
# verts3.append((x[16282], y_values[2], 5))   # 添加底部点
# poly3 = Poly3DCollection([verts3], facecolors='r', alpha=0.4)  # 创建三维多边形
# ax.add_collection3d(poly3)  # 添加到三维图中
#
# # 设置 y 轴刻度和标签
# ax.set_yticks(y_values)  # 设置刻度位置
# ax.set_yticklabels(y_labels)  # 设置刻度标签
#
# #设置三维图图形区域背景颜色（r,g,b,a）
# ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
# ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
# ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
#
# # 设置标签
# # ax.set_xlabel('X Axis')
# # ax.set_ylabel('Y Axis')
# # ax.set_zlabel('Z Axis')
#
# # 设置视角
# ax.view_init(elev=10, azim=-79)
#
# # 设置图形范围
# # ax.set_xlim([-5, 5])
# ax.set_ylim([min(y_values) - 0, max(y_values) + 0])  # 扩展 y 轴范围以显示标签
# ax.set_zlim([0, 5])
#
# # 调整横纵比
# ax.set_box_aspect([2, 1.5, 1])  # 设置 x、y、z 轴的比例
# plt.rc('font', size=8,family='Times New Roman')
#
# # 显示图形
# plt.show()

##########################################兴奋/抑制_权重控制_数据绘图1############################################

##########################################兴奋/抑制_权重控制_数据绘图2############################################
# from mpl_toolkits.mplot3d.art3d import Poly3DCollection
# from matplotlib.patches import Patch
#
# file_path = "D:\信息与能量研究方向\论文\基于随机热力学的神经形态电路\神经元.xlsx"
# # 使用 usecols 选择列，skiprows 跳过不需要的行
# data = pd.read_excel(file_path, usecols='A:I', skiprows=76, nrows=20000, sheet_name='兴奋抑制')  # 从第2行起，读取A列到C列，5行数据
# # print(data)
# # 创建数据
# x = data['time1']  # x 轴数据
# z1_1 = data['Vout_ctrl1']
# z1_2 = data['Vout1']
# z2_1 = data['Vout_ctrl2']
# z2_2 = data['Vout2']
# z3_1 = data['Vout_ctrl3']
# z3_2 = data['Vout3']
# z3 = np.linspace(5, 5, 100)
#
# # 定义多个 y 值（分层）和对应的标签
# y_values = [0, 1, 2]  # 每个平面在 y 轴上的位置
# y_labels = ['100', '500', '1000']  # 自定义标签
#
# # 创建三维图形
# fig = plt.figure(figsize=(3, 2))
# ax = fig.add_subplot(111, projection='3d')
# # 绘制每个平面
# # for y, label in zip(y_values, y_labels):
# # 绘制第一条曲线
# ax.plot(x, [y_values[0]] * len(x), z1_1, color='#004496', alpha=0.2)  # 曲线
# # 绘制第二条曲线
# ax.plot(x, [y_values[0]] * len(x), z1_2, color='#374C6D', alpha=0.2)  # 曲线
# # 填充第一条曲线以下的区域
# verts1 = list(zip(x, [y_values[0]] * len(x), z1_1))  # 创建顶点
# verts1.append((x[19999], y_values[0], 0))  # 添加底部点
# verts1.append((x[0], y_values[0], 0))   # 添加底部点
# poly1 = Poly3DCollection([verts1], facecolors='#004496', alpha=0.2)  # 创建三维多边形
# ax.add_collection3d(poly1)  # 添加到三维图中
# # 填充第二条曲线以下的区域
# verts2 = list(zip(x, [y_values[0]] * len(x), z1_2))  # 创建顶点
# verts2.append((x[19999], y_values[0], 0))  # 添加底部点
# verts2.append((x[0], y_values[0], 0))   # 添加底部点
# poly2 = Poly3DCollection([verts2], facecolors='#374C6D', alpha=0.2)  # 创建三维多边形
# ax.add_collection3d(poly2)  # 添加到三维图中
# # 填充尖峰区间
# verts3 = list(zip(x[4038:7287], [y_values[0]] * len(x[4038:7287]), z3))  # 创建顶点
# verts3.append((x[4038], y_values[0], 0))  # 添加底部点
# verts3.append((x[7287], y_values[0], 0))   # 添加底部点
# verts3.append((x[7287], y_values[0], 5))   # 添加底部点
# poly3 = Poly3DCollection([verts3], facecolors='r', alpha=0.4)  # 创建三维多边形
# ax.add_collection3d(poly3)  # 添加到三维图中
#
# # 绘制第一条曲线
# ax.plot(x, [y_values[1]] * len(x), z2_1, color='#004496', alpha=0.2)  # 曲线
# # 绘制第二条曲线
# ax.plot(x, [y_values[1]] * len(x), z2_2, color='#374C6D', alpha=0.2)  # 曲线
# # 填充第一条曲线以下的区域
# verts1 = list(zip(x, [y_values[1]] * len(x), z2_1))  # 创建顶点
# verts1.append((x[19999], y_values[1], 0))  # 添加底部点
# verts1.append((x[0], y_values[1], 0))   # 添加底部点
# poly1 = Poly3DCollection([verts1], facecolors='#004496', alpha=0.2)  # 创建三维多边形
# ax.add_collection3d(poly1)  # 添加到三维图中
# # 填充第二条曲线以下的区域
# verts2 = list(zip(x, [y_values[1]] * len(x), z2_2))  # 创建顶点
# verts2.append((x[19999], y_values[1], 0))  # 添加底部点
# verts2.append((x[0], y_values[1], 0))   # 添加底部点
# poly2 = Poly3DCollection([verts2], facecolors='#374C6D', alpha=0.2)  # 创建三维多边形
# ax.add_collection3d(poly2)  # 添加到三维图中
# # 填充尖峰区间
# verts3 = list(zip(x[9038:10899], [y_values[1]] * len(x[9038:10899]), z3))  # 创建顶点
# verts3.append((x[9038], y_values[1], 0))  # 添加底部点
# verts3.append((x[10899], y_values[1], 0))   # 添加底部点
# verts3.append((x[10899], y_values[1], 5))   # 添加底部点
# poly3 = Poly3DCollection([verts3], facecolors='r', alpha=0.4)  # 创建三维多边形
# ax.add_collection3d(poly3)  # 添加到三维图中
#
# # 绘制第一条曲线
# ax.plot(x, [y_values[2]] * len(x), z3_1, color='#004496', alpha=0.2)  # 曲线
# # 绘制第二条曲线
# ax.plot(x, [y_values[2]] * len(x), z3_2, color='#374C6D', alpha=0.2)  # 曲线
# # 填充第一条曲线以下的区域
# verts1 = list(zip(x, [y_values[2]] * len(x), z3_1))  # 创建顶点
# verts1.append((x[19999], y_values[2], 0))  # 添加底部点
# verts1.append((x[0], y_values[2], 0))   # 添加底部点
# poly1 = Poly3DCollection([verts1], facecolors='#004496', alpha=0.2)  # 创建三维多边形
# ax.add_collection3d(poly1)  # 添加到三维图中
# # 填充第二条曲线以下的区域
# verts2 = list(zip(x, [y_values[2]] * len(x), z3_2))  # 创建顶点
# verts2.append((x[19999], y_values[2], 0))  # 添加底部点
# verts2.append((x[0], y_values[2], 0))   # 添加底部点
# poly2 = Poly3DCollection([verts2], facecolors='#374C6D', alpha=0.2)  # 创建三维多边形
# ax.add_collection3d(poly2)  # 添加到三维图中
# # 填充尖峰区间
# verts3 = list(zip(x[14038:15185], [y_values[2]] * len(x[14038:15185]), z3))  # 创建顶点
# verts3.append((x[14038], y_values[2], 0))  # 添加底部点
# verts3.append((x[15185], y_values[2], 0))   # 添加底部点
# verts3.append((x[15185], y_values[2], 5))   # 添加底部点
# poly3 = Poly3DCollection([verts3], facecolors='r', alpha=0.4)  # 创建三维多边形
# ax.add_collection3d(poly3)  # 添加到三维图中
#
# # 设置 y 轴刻度和标签
# ax.set_yticks(y_values)  # 设置刻度位置
# ax.set_yticklabels(y_labels)  # 设置刻度标签
#
# #设置三维图图形区域背景颜色（r,g,b,a）
# ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
# ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
# ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
#
# # 设置标签
# # ax.set_xlabel('X Axis')
# # ax.set_ylabel('Y Axis')
# # ax.set_zlabel('Z Axis')
#
# # 设置视角
# ax.view_init(elev=10, azim=-79)
#
# # 设置图形范围
# # ax.set_xlim([-5, 5])
# ax.set_ylim([min(y_values) - 0, max(y_values) + 0])  # 扩展 y 轴范围以显示标签
# ax.set_zlim([0, 5])
#
# # 调整横纵比
# ax.set_box_aspect([2, 1.5, 1])  # 设置 x、y、z 轴的比例
# plt.rc('font', size=8,family='Times New Roman')
# # 显示图例
# # 创建代理对象用于图例
# proxy1 = Patch(color='#374C6D', alpha=0.2, label='$V_{\mathrm{mem}}$')
# proxy2 = Patch(color='#004496', alpha=0.2, label='$V_{\mathrm{off}}$')
# proxy3 = Patch(color='r', alpha=0.4, label='$t_{\mathrm{in}}$')
# # ax.legend(handles=[proxy1, proxy2], loc='right')
# # 显示图例，放置在图像之外
# legend = ax.legend(handles=[proxy1, proxy2, proxy3], loc='right', bbox_to_anchor=(1.55, 0.5), borderaxespad=0., fontsize=9)
#
# # 调整布局，为图例腾出空间
# plt.tight_layout()
# # 显示图形
# plt.show()

##########################################兴奋/抑制_权重控制_数据绘图2############################################

##########################################兴奋/抑制实例############################################
# Vout_E1 = 0.0
# Vout_E2 = 0.0
# Vout_E3 = 0.0
# E0 = 5.0
# # delta_E0 = -1.0
#
# tint = 10
# T = 1000000
# Ntot = int(T / tint)
# t = np.linspace(0, T, tint)
# Vth = 5.0
# flag_E1 = 0
# flag_E2 = 0
# flag_E3 = 0
# flag_break = 0
#
# output_E1 = np.zeros(Ntot)
# output_E2 = np.zeros(Ntot)
# output_E3 = np.zeros(Ntot)
# Jout2 = np.zeros(Ntot)
# Jout3 = np.zeros(Ntot)
# Vg_E = np.zeros(Ntot)
# time = np.zeros(Ntot)
# flag = np.zeros(Ntot)
# diss_MOS2 = np.zeros(Ntot)
# energy_MOS2 = np.zeros(Ntot)
# diss_GND2 = np.zeros(Ntot)
# energy_GND2 = np.zeros(Ntot)
# diss_MOS_R2 = 0.0
# diss_GND_R2 = 0.0
# diss_MOS3 = np.zeros(Ntot)
# energy_MOS3 = np.zeros(Ntot)
# diss_GND3 = np.zeros(Ntot)
# energy_GND3 = np.zeros(Ntot)
# diss_MOS_R3 = 0.0
# diss_GND_R3 = 0.0
# Jout_post = np.zeros(Ntot)
#
# Jin1 = np.zeros(Ntot)
# Jin2 = np.zeros(Ntot)
# Jin3 = np.zeros(Ntot)
# J_GND2 = np.zeros(Ntot)
# J_GND3 = np.zeros(Ntot)
# start_t = 20000
# delta_t = 3000
# Jin1[start_t - 1000:start_t + 0] = 0.05    #控制Vctrl
# Jin2[start_t:start_t + 3000] = 0.05    #阈值Vth
# Jin3[start_t + 5000:start_t + 8000] = 0.05    #输入后置Vfire
# for i in range(Ntot):
#     output_E1[i] = Vout_E1
#     output_E2[i] = Vout_E2
#     output_E3[i] = Vout_E3
#     _, _, Vout_E1, _, flag_E1, _, _ = NEURON(Jin1[i], Vout_E1, Vth, E0, flag_E1, 1, 0.0)
#     Jout2[i], J_GND2[i], Vout_E2, Vg_E_R2, flag_E2, diss_MOS2[i], diss_GND2[i] = NEURON(Jin2[i], Vout_E2, Vth, E0, flag_E2, 1, output_E1[i])
#     Jout3[i], J_GND3[i], Vout_E3, Vg_E_R3, flag_E3, diss_MOS3[i], diss_GND3[i] = NEURON(Jin3[i], Vout_E3, Vth, E0, flag_E3, 1, 0.0)
#     # Vg_E[i] = Vg_E_R
#     flag[i] = flag_E3
#     diss_MOS_R2 = diss_MOS2[i] + diss_MOS_R2
#     energy_MOS2[i] = diss_MOS_R2
#     diss_GND_R2 = diss_GND2[i] + diss_GND_R2
#     energy_GND2[i] = diss_GND_R2
#     time[i] = i * tint
#     if(i > 0):
#         if(flag[i] == 1)&(output_E3[i] >= output_E2[i]):
#             Jout_post[i] = 0.05
#
# print(diss_MOS_R2, diss_GND_R2, diss_MOS_R2 + diss_GND_R2)
#
# plt.figure(1, figsize=(15, 10), dpi=300)
# plt.rcParams['axes.formatter.min_exponent'] = 1  # 将最小指数更改为1
# plt.rc('font', size=16, family='Times New Roman')  # 将字体大小更改为12
# plt.rcParams['xtick.direction'] = 'in'  # 将x周的刻度线方向设置向内
# plt.rcParams['ytick.direction'] = 'in'  # 将y轴的刻度方向设置向内
# plt.subplot(2, 2, 1)
# plt.plot(time, Jin1, color='cornflowerblue')
# plt.plot(time, Jin2, color='orange')
# plt.plot(time, Jin3, color='black')
# plt.xlabel("Time")
# plt.ylabel("Jin")
# plt.subplot(2, 2, 2)
# plt.plot(time, output_E1, color='cornflowerblue')
# plt.plot(time, output_E2, color='orange')
# plt.plot(time, output_E3, color='black')
# plt.xlabel("Time")
# plt.ylabel("Vout")
# plt.subplot(2, 2, 3)
# plt.plot(time, Jout_post, color='orange')
# # plt.plot(time, J_GND, color='cornflowerblue')
# # print(max(Jout))
# plt.xlabel("Time")
# plt.ylabel("Jout_post")
# plt.subplot(2, 2, 4)
# plt.plot(time, energy_MOS2, color='orange')
# plt.plot(time, energy_GND2, color='cornflowerblue')
# plt.plot(time, energy_MOS2 + energy_GND2, color='black')
# plt.xlabel("Time")
# plt.ylabel("diss")
#
# plt.tight_layout()
# plt.savefig('兴奋抑制实例.svg', format='svg')


##########################################兴奋/抑制实例############################################

##########################################NOT############################################
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
# flag_break = 0
#
# output_E1 = np.zeros(Ntot)
# output_E2 = np.zeros(Ntot)
# Jout = np.zeros(Ntot)
# Vg_E = np.zeros(Ntot)
# time = np.zeros(Ntot)
# flag = np.zeros(Ntot)
# diss_MOS1 = np.zeros(Ntot)
# diss_MOS2 = np.zeros(Ntot)
# energy_MOS = np.zeros(Ntot)
# diss_GND1 = np.zeros(Ntot)
# diss_GND2 = np.zeros(Ntot)
# energy_GND = np.zeros(Ntot)
# diss_MOS_R1 = 0.0
# diss_GND_R1 = 0.0
# diss_MOS_R2 = 0.0
# diss_GND_R2 = 0.0
# Jout_post = np.zeros(Ntot)
#
# Jin1 = np.zeros(Ntot)
# Jin2 = np.zeros(Ntot)
# J_GND = np.zeros(Ntot)
#
# Jin1[25000:25486] = 0.05    #控制
# Jin2[25000:25486] = 0.05    #输入输出
# Jin2[75000:75486] = 0.05    #输入输出
# for i in range(Ntot):
#     output_E1[i] = Vout_E1
#     output_E2[i] = Vout_E2
#     _, _, Vout_E1, _, flag_E1, diss_MOS1[i], diss_GND1[i] = NEURON(Jin1[i], Vout_E1, Vth, E0, flag_E1, 1, 0.0)
#     Jout[i], J_GND[i], Vout_E2, Vg_E_R, flag_E2, diss_MOS2[i], diss_GND2[i] = NEURON(Jin2[i], Vout_E2, Vth, E0, flag_E2, 1, 0.0)
#     # Vg_E[i] = Vg_E_R
#     flag[i] = flag_E2
#     diss_MOS_R1 = diss_MOS1[i] + diss_MOS_R1
#     diss_GND_R1 = diss_GND1[i] + diss_GND_R1
#     diss_MOS_R2 = diss_MOS2[i] + diss_MOS_R2
#     diss_GND_R2 = diss_GND2[i] + diss_GND_R2
#     energy_MOS[i] = diss_MOS_R1 + diss_MOS_R2
#     energy_GND[i] = diss_GND_R1 + diss_GND_R2
#     time[i] = i * tint
#     if(i > 0):
#         if(flag[i - 1] == 0)&(flag[i] == 1)&(output_E2[i] > output_E1[i]):
#             Jout_post[i : i + 486] = 0.05
#
# print(diss_MOS_R1, diss_GND_R1, diss_MOS_R1 + diss_GND_R1)
# print(diss_MOS_R2, diss_GND_R2, diss_MOS_R2 + diss_GND_R2)
# print(diss_MOS_R1 + diss_MOS_R2, diss_GND_R1 + diss_GND_R2, diss_MOS_R1 + diss_GND_R1 + diss_MOS_R2 + diss_GND_R2)
#
# plt.figure(1, figsize=(10, 8), dpi=300)
# plt.rcParams['axes.formatter.min_exponent'] = 1  # 将最小指数更改为1
# plt.rc('font', size=14, family='Times New Roman')  # 将字体大小更改为16
# plt.rcParams['xtick.direction'] = 'in'  # 将x周的刻度线方向设置向内
# plt.rcParams['ytick.direction'] = 'in'  # 将y轴的刻度方向设置向内
#
# subplot(4, 1, 1)
# plot(time, Jin1, color='#4c6792', linewidth='2')
# plt.xlabel("Time ($β\hbar$)")
# plt.ylabel("Jin_ctrl")
# # plt.text(-0.15e+7, 5, '(a)', fontsize=16, fontproperties='Times New Roman')
#
# subplot(4, 1, 2)
# plot(time, Jin2, color='orange', linewidth='2')
# plt.xlabel("Time ($β\hbar$)")
# plt.ylabel("Jin_input1")
# # plt.text(-0.15e+7, 5, '(b)', fontsize=16, fontproperties='Times New Roman')
#
# subplot(4, 1, 3)
# plot(time, output_E1, color='#4c6792', linewidth='2')
# plot(time, output_E2, linestyle='--', color='orange', linewidth='2')
# plt.xlabel("Time ($β\hbar$)")
# plt.ylabel("Vout")
# # plt.text(-0.15e+7, 4.5, '(c)', fontsize=16, fontproperties='Times New Roman')
#
# subplot(4, 1, 4)
# plot(time, Jout_post, color='#4c6792', linewidth='2')
# plt.xlabel("Time ($β\hbar$)")
# plt.ylabel("Jout_post")
# # plt.text(-0.15e+7, 4.5, '(c)', fontsize=16, fontproperties='Times New Roman')
#
# # plt.figure(1, figsize=(15, 10), dpi=300)
# # plt.rcParams['axes.formatter.min_exponent'] = 1  # 将最小指数更改为1
# # plt.rc('font', size=16, family='Times New Roman')  # 将字体大小更改为12
# # plt.rcParams['xtick.direction'] = 'in'  # 将x周的刻度线方向设置向内
# # plt.rcParams['ytick.direction'] = 'in'  # 将y轴的刻度方向设置向内
# # plt.subplot(2, 2, 1)
# # plt.plot(time, Jin1, color='cornflowerblue')
# # plt.plot(time, Jin2, color='orange')
# # plt.xlabel("Time")
# # plt.ylabel("Jin")
# # plt.subplot(2, 2, 2)
# # plt.plot(time, output_E1, color='cornflowerblue')
# # plt.plot(time, output_E2, color='orange')
# # plt.xlabel("Time")
# # plt.ylabel("Vout")
# # plt.subplot(2, 2, 3)
# # plt.plot(time, Jout_post, color='orange')
# # # plt.plot(time, J_GND, color='cornflowerblue')
# # plt.xlabel("Time")
# # plt.ylabel("Jout_post")
# # plt.subplot(2, 2, 4)
# # plt.plot(time, energy_MOS, color='orange')
# # plt.plot(time, energy_GND, color='cornflowerblue')
# # plt.plot(time, energy_MOS + energy_GND, color='black')
# # plt.xlabel("Time")
# # plt.ylabel("diss")
#
# plt.tight_layout()
# plt.savefig('logic_NOT.svg', format='svg')
# data = {'time': time, 'Jin1': Jin1, 'Jin2': Jin2, 'Vmem1': output_E1, 'Vmem2': output_E2, 'Jout': Jout_post}
# df = pd.DataFrame(data)
# df.to_excel('output_logic.xlsx', index=False)
##########################################NOT############################################

##########################################尖峰检测############################################
# Vout_E = 0.0
# Vout_I = 0.0
# E0 = 6.0
# delta_E0 = 0.0
#
# tint = 10
# T = 1000000
# Ntot = int(T / tint)
# t = np.linspace(0, T, tint)
# Vth = 6.0
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
# delta_t = 2429
# Jin[start_t:start_t + delta_t] = 0.05
# t_90 = 22668
# diff = np.zeros(t_90 - start_t - delta_t)
# cnt_diff = np.zeros(t_90 - start_t - delta_t)
# j = 0
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
#     if (i > start_t + delta_t) & (i <= t_90):
#         # print(i, output_E[i - 1] - output_E[i])
#         diff[j] = (output_E[i - 1] - output_E[i]) * 0.1
#         cnt_diff[j] = i
#         j = j + 1
#
# data = {'cnt_diff': cnt_diff, 'diff': diff}
# df = pd.DataFrame(data)
# df.to_excel('output.xlsx', index=False)

##########################################尖峰检测############################################
