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
Jout_post = np.zeros(Ntot)

Jin = np.zeros(Ntot)
J_GND = np.zeros(Ntot)
start_t = 20000
delta_t = 500
Jin[start_t:start_t + delta_t] = 0.05
# Jin[20000:21000] = 0.05
# Jin[30000:31000] = 0.05
# Jin[40000:41000] = 0.05

for i in range(Ntot):
    output_E[i] = Vout_E
    output_I[i] = Vout_I
    Jout[i], J_GND[i], Vout_E, Vg_E_R, flag_E, diss_MOS[i], diss_GND[i] = NEURON(Jin[i], Vout_E, Vth, E0, flag_E, 1, 0.0)
    Vg_E[i] = Vg_E_R
    flag[i] = flag_E
    diss_MOS_R = diss_MOS[i] + diss_MOS_R
    energy_MOS[i] = diss_MOS_R
    diss_GND_R = diss_GND[i] + diss_GND_R
    energy_GND[i] = diss_GND_R
    time[i] = i * tint

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
plt.xlabel("Time")
plt.ylabel("J")
plt.subplot(2, 2, 4)
plt.plot(time, energy_MOS, color='orange')
plt.plot(time, energy_GND, color='cornflowerblue')
plt.plot(time, energy_MOS + energy_GND, color='black')
plt.xlabel("Time")
plt.ylabel("diss")

plt.tight_layout()
plt.savefig('logic.svg', format='svg')

##########################################test############################################

##########################################AND############################################单位时间和总时间均需扩大十倍以避免干扰
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
# Jin1 = np.zeros(Ntot)
# Jin2 = np.zeros(Ntot)
# Jin1[60000:60219] = 0.05
# Jin1[80000:80219] = 0.05
# Jin2[40000:40219] = 0.05
# Jin2[80000:80219] = 0.05
# J_GND = np.zeros(Ntot)
#
#
# for i in range(Ntot):
#     output_E[i] = Vout_E
#     output_I[i] = Vout_I
#     Jout[i], J_GND[i], Vout_E, Vg_E_R, flag_E, diss_MOS[i], diss_GND[i] = NEURON(Jin1[i] + Jin2[i], Vout_E, Vth, E0, flag_E, 1, 0.0)
#     # J_I[i], Vout_I, Vg_I_R, flag_I = NEURON(Jin[i], Vout_I, Vth, E0, flag_I, 0)
#     Vg_E[i] = Vg_E_R
#     flag[i] = flag_E
#     diss_MOS_R = diss_MOS[i] + diss_MOS_R
#     energy_MOS[i] = diss_MOS_R
#     diss_GND_R = diss_GND[i] + diss_GND_R
#     energy_GND[i] = diss_GND_R
#     time[i] = i * tint
#     if(i > 0):
#         if(flag[i - 1] == 0)&(flag[i] == 1):
#             Jout_post[i:i + 219] = 0.05
# print(diss_MOS_R, diss_GND_R, diss_MOS_R + diss_GND_R)
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
# plt.ylabel("Jin1")
# # plt.text(-0.15e+7, 5, '(a)', fontsize=16, fontproperties='Times New Roman')
#
# subplot(4, 1, 2)
# plot(time, Jin2, color='#4c6792', linewidth='2')
# plt.xlabel("Time ($β\hbar$)")
# plt.ylabel("Jin2")
# # plt.text(-0.15e+7, 5, '(b)', fontsize=16, fontproperties='Times New Roman')
#
# subplot(4, 1, 3)
# plot(time, output_E, color='#4c6792', linewidth='2')
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
#
# plt.tight_layout()
# plt.savefig('logic_AND.svg', format='svg')
# data = {'time': time, 'Jin1': Jin1, 'Jin2': Jin2, 'Vmem': output_E, 'Jout': Jout_post}
# df = pd.DataFrame(data)
# df.to_excel('output_logic.xlsx', index=False)
##########################################AND############################################

##########################################OR############################################单位时间和总时间均需扩大十倍以避免干扰
# Vout_E = 0.0
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

# Jin1 = np.zeros(Ntot)
# Jin2 = np.zeros(Ntot)
# Jin1[60000:60486] = 0.05
# Jin1[80000:80486] = 0.05
# Jin2[40000:40486] = 0.05
# Jin2[80000:80486] = 0.05
# J_GND = np.zeros(Ntot)
#
# for i in range(Ntot):
#     output_E[i] = Vout_E
#     output_I[i] = Vout_I
#     Jout[i], J_GND[i], Vout_E, Vg_E_R, flag_E, diss_MOS[i], diss_GND[i] = NEURON(Jin1[i] + Jin2[i], Vout_E, Vth, E0, flag_E, 1, 0.0)
#     # J_I[i], Vout_I, Vg_I_R, flag_I = NEURON(Jin[i], Vout_I, Vth, E0, flag_I, 0)
#     Vg_E[i] = Vg_E_R
#     flag[i] = flag_E
#     diss_MOS_R = diss_MOS[i] + diss_MOS_R
#     energy_MOS[i] = diss_MOS_R
#     diss_GND_R = diss_GND[i] + diss_GND_R
#     energy_GND[i] = diss_GND_R
#     time[i] = i * tint
#     if (i > 0):
#         if (flag[i - 1] == 0) & (flag[i] == 1):
#             Jout_post[i:i + 486] = 0.05
# print(diss_MOS_R, diss_GND_R, diss_MOS_R + diss_GND_R)
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
# plt.ylabel("Jin1")
# # plt.text(-0.15e+7, 5, '(a)', fontsize=16, fontproperties='Times New Roman')
#
# subplot(4, 1, 2)
# plot(time, Jin2, color='#4c6792', linewidth='2')
# plt.xlabel("Time ($β\hbar$)")
# plt.ylabel("Jin2")
# # plt.text(-0.15e+7, 5, '(b)', fontsize=16, fontproperties='Times New Roman')
#
# subplot(4, 1, 3)
# plot(time, output_E, color='#4c6792', linewidth='2')
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
# plt.tight_layout()
# plt.savefig('logic_OR.svg', format='svg')
# data = {'time': time, 'Jin1': Jin1, 'Jin2': Jin2, 'Vmem': output_E, 'Jout': Jout_post}
# df = pd.DataFrame(data)
# df.to_excel('output_logic.xlsx', index=False)
##########################################OR############################################

##########################################logic绘图_汇总############################################
#
# file_path = "D:\信息与能量研究方向\论文\基于随机热力学的神经形态电路\神经元.xlsx"
# # 使用 usecols 选择列，skiprows 跳过不需要的行
# data = pd.read_excel(file_path, usecols='A:S', skiprows=1, nrows=100001, sheet_name='逻辑')  # 从第2行起，读取A列到C列，5行数据
# # print(data)
#
# plt.figure(1, figsize=(5.8, 1.8), dpi=300)
# plt.rcParams['axes.formatter.min_exponent'] = 1  # 将最小指数更改为1
# plt.rc('font', size=8, family='Times New Roman')  # 将字体大小更改为12
# plt.rcParams['xtick.direction'] = 'in'  # 将x周的刻度线方向设置向内
# plt.rcParams['ytick.direction'] = 'in'  # 将y轴的刻度方向设置向内
#
# time = data['time']
#
# plt.subplot(4, 4, 1)
# plt.plot(time * 1e-6, data['Jin1_AND'], color='#374C6D', linewidth=1)
# plt.xticks([])
# plt.xlim([0.0, 1.0])
# plt.subplot(4, 4, 5)
# plt.plot(time * 1e-6, data['Jin2_AND'], color='#374C6D', linewidth=1)
# plt.xticks([])
# plt.xlim([0.0, 1.0])
# plt.subplot(4, 4, 9)
# plt.plot(time * 1e-6, data['Vmem_AND'], color='#374C6D', linewidth=1)
# plt.xticks([])
# plt.xlim([0.0, 1.0])
# plt.subplot(4, 4, 13)
# plt.plot(time * 1e-6, data['Jout_AND'], color='#374C6D', linewidth=1)
# plt.xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
# plt.xlim([0.0, 1.0])
#
# plt.subplot(4, 4, 2)
# plt.plot(time * 1e-6, data['Jin1_OR'], color='#374C6D', linewidth=1)
# plt.xticks([])
# plt.yticks([])
# plt.xlim([0.0, 1.0])
# plt.subplot(4, 4, 6)
# plt.plot(time * 1e-6, data['Jin2_OR'], color='#374C6D', linewidth=1)
# plt.xticks([])
# plt.yticks([])
# plt.xlim([0.0, 1.0])
# plt.subplot(4, 4, 10)
# plt.plot(time * 1e-6, data['Vmem_OR'], color='#374C6D', linewidth=1)
# plt.xticks([])
# plt.yticks([])
# plt.xlim([0.0, 1.0])
# plt.subplot(4, 4, 14)
# plt.plot(time * 1e-6, data['Jout_OR'], color='#374C6D', linewidth=1)
# plt.yticks([])
# plt.xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
# plt.xlim([0.0, 1.0])
#
# plt.subplot(4, 4, 3)
# plt.plot(time * 1e-6, data['Jin1_NOT'], color='#374C6D', linewidth=1)
# plt.xticks([])
# plt.yticks([])
# plt.xlim([0.0, 1.0])
# plt.subplot(4, 4, 7)
# plt.plot(time * 1e-6, data['Jin2_NOT'], color='#374C6D', linewidth=1)
# plt.xticks([])
# plt.yticks([])
# plt.xlim([0.0, 1.0])
# plt.subplot(4, 4, 11)
# plt.plot(time * 1e-6, data['Vmem1_NOT'], color='#374C6D', linewidth=1)
# plt.plot(time * 1e-6, data['Vmem2_NOT'], linestyle='--', color='orange', linewidth=1, alpha=1.0)
# plt.xticks([])
# plt.yticks([])
# plt.xlim([0.0, 1.0])
# plt.subplot(4, 4, 15)
# plt.plot(time * 1e-6, data['Jout_NOT'], color='#374C6D', linewidth=1)
# plt.yticks([])
# plt.xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
# plt.xlim([0.0, 1.0])
#
# plt.subplot(4, 4, 4)
# plt.plot(time * 1e-6, data['Jin2_XOR'], color='#374C6D', linewidth=1)
# plt.xticks([])
# plt.yticks([])
# plt.xlim([0.0, 1.0])
# plt.subplot(4, 4, 8)
# plt.plot(time * 1e-6, data['Jin1_XOR'], color='#374C6D', linewidth=1)
# plt.xticks([])
# plt.yticks([])
# plt.xlim([0.0, 1.0])
# plt.subplot(4, 4, 12)
# plt.plot(time * 1e-6, data['Vmem2_XOR'], color='#374C6D', linewidth=1)
# plt.plot(time * 1e-6, data['Vmem1_XOR'], linestyle='--', color='orange', linewidth=1, alpha=1.0)
# plt.xticks([])
# plt.yticks([])
# plt.xlim([0.0, 1.0])
# plt.subplot(4, 4, 16)
# plt.plot(time * 1e-6, data['Jout_XOR'], color='#374C6D', linewidth=1)
# plt.yticks([])
# plt.xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
# plt.xlim([0.0, 1.0])
#
#
#
# plt.tight_layout()
# # plt.savefig('logic.svg', format='svg')
# plt.show()
##########################################logic绘图_汇总############################################

##########################################logic绘图_能耗对比############################################
#
# input_bit = ('AND', 'OR', 'NOT', 'XOR')
# W_cmos = [5343, 4573, 2282, 13616]
# W_neuron = [42.19, 92.63, 193.4, 137.9]
# #
# # M_max = [Mmax3, Mmax2, Mmax1]
# # M_min = [Mmin3, Mmin2, Mmin1]
#
# # plt.rc('font',family = 'Times New Roman')
# # plt.figure(1, dpi=300)
# # plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
#
#
# x0 = np.arange(-0.15, 0.45, 0.001)
# y0 = 5343 * np.ones(600)
# x1 = np.arange(0.15, 0.45, 0.001)
# y1 = 42.19 * np.ones(301)
#
# x2 = np.arange(0.85, 1.45, 0.001)
# y2 = 4573 * np.ones(600)
# x3 = np.arange(1.15, 1.45, 0.001)
# y3 = 92.63 * np.ones(301)
#
# x4 = np.arange(1.85, 2.45, 0.001)
# y4 = 2282 * np.ones(601)
# x5 = np.arange(2.15, 2.45, 0.001)
# y5 = 193.4 * np.ones(301)
#
# x6 = np.arange(2.85, 3.45, 0.001)
# y6 = 13616 * np.ones(601)
# x7 = np.arange(3.15, 3.45, 0.001)
# y7 = 137.9 * np.ones(301)
#
#
#
# plt.figure(1, figsize=(3.2, 2.0), dpi=600)
# # plt.figure(1, dpi=300)
# plt.rc('font',family = 'Times New Roman')
# # plt.subplot(1, 2, 1)
# plt.rcParams['xtick.direction'] = 'in'#将x周的刻度线方向设置向内
# plt.rcParams['ytick.direction'] = 'in'#将y轴的刻度方向设置向内
# bar_width = 0.3  # 条形宽度
# index_max = np.arange(len(input_bit))  # max条形图的横坐标
# index_min = index_max + bar_width  # min条形图的横坐标
# # plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
# plt.bar(index_max, height=W_cmos, width=bar_width, color='#374C6D', label='CMOS')
# plt.bar(index_min, height=W_neuron, width=bar_width, color='#a3abbd', label='NC')
# plt.legend(loc='upper left',fontsize=8, ncol=2)
# # plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), borderaxespad=0., fontsize=8, ncol=2)
# plt.xticks(index_max + bar_width / 2, input_bit,fontproperties='Times New Roman', fontsize=9)
# plt.yticks(fontproperties='Times New Roman', fontsize=8)
# # plt.ylabel("Energy consumption($kT$)",fontproperties='Times New Roman', fontsize=14)
#
# plt.plot(x0, y0, color='orange', linewidth=1.5)
# plt.plot(x1, y1, color='orange', linewidth=1.5)
# plt.annotate('', xy=(0.3, 38), xytext=(0.3, 5843), arrowprops={'arrowstyle': '<->', 'color': 'orange'})
# # plt.text(0.33, 400, '99.21%', color='#c00000', fontproperties='Times New Roman', fontsize=9)
#
# plt.plot(x2, y2, color='orange', linewidth=1.5)
# plt.plot(x3, y3, color='orange', linewidth=1.5)
# plt.annotate('', xy=(1.3, 82.63), xytext=(1.3, 5073), arrowprops={'arrowstyle': '<->', 'color': 'orange'})
# # plt.text(1.33, 600, '97.97%', color='#c00000', fontproperties='Times New Roman', fontsize=9)
#
# plt.plot(x4, y4, color='orange', linewidth=1.5)
# plt.plot(x5, y5, color='orange', linewidth=1.5)
# plt.annotate('', xy=(2.3, 173.4), xytext=(2.3, 2482), arrowprops={'arrowstyle': '<->', 'color': 'orange'})
# # plt.text(2.33, 550, '91.52%', color='#c00000', fontproperties='Times New Roman', fontsize=9)
#
# plt.plot(x6, y6, color='orange', linewidth=1.5)
# plt.plot(x7, y7, color='orange', linewidth=1.5)
# plt.annotate('', xy=(3.3, 122.9), xytext=(3.3, 15000), arrowprops={'arrowstyle': '<->', 'color': 'orange'})
# # plt.text(3.33, 1200, '98.99%', color='#c00000', fontproperties='Times New Roman', fontsize=9)
#
# plt.yscale('log')
#
# plt.tight_layout()
# plt.savefig('logic_能耗对比.jpg', format='jpg', bbox_inches='tight')

##########################################logic绘图_能耗对比############################################

##########################################logic绘图_时延对比############################################
#
# input_bit = ('AND', 'OR', 'NOT', 'XOR', 'NC')
# W_time = [650000, 650000, 450000, 900000, 55020]
# #
# # M_max = [Mmax3, Mmax2, Mmax1]
# # M_min = [Mmin3, Mmin2, Mmin1]
#
# # plt.rc('font',family = 'Times New Roman')
# # plt.figure(1, dpi=300)
# # plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
#
#
# x0 = np.arange(0.0, 1.0, 0.001)
# y0 = W_time[0] * np.ones(1000)
# # x1 = np.arange(0.15, 0.45, 0.001)
# # y1 = 42.19 * np.ones(301)
#
# x2 = np.arange(1.0, 2.0, 0.001)
# y2 = W_time[1] * np.ones(1000)
# # x3 = np.arange(1.15, 1.45, 0.001)
# # y3 = 92.63 * np.ones(301)
#
# x4 = np.arange(2.0, 3.0, 0.001)
# y4 = W_time[2] * np.ones(1000)
# # x5 = np.arange(2.15, 2.45, 0.001)
# # y5 = 193.4 * np.ones(301)
#
# x6 = np.arange(3.0, 4.0, 0.001)
# y6 = W_time[3] * np.ones(1000)
# # x7 = np.arange(3.15, 3.45, 0.001)
# # y7 = 137.9 * np.ones(301)
#
# x8 = np.arange(0.0, 4.5, 0.001)
# y8 = W_time[4] * np.ones(4500)
# x9 = np.arange(4.0, 4.5, 0.001)
# y9 = W_time[4] * np.ones(500)
# plt.figure(1, figsize=(2.6, 2.04), dpi=600)
# # plt.figure(1, dpi=300)
# plt.rc('font',family = 'Times New Roman')
# # plt.subplot(1, 2, 1)
# plt.rcParams['xtick.direction'] = 'in'#将x周的刻度线方向设置向内
# plt.rcParams['ytick.direction'] = 'in'#将y轴的刻度方向设置向内
# bar_width = 0.5  # 条形宽度
# index_max = np.arange(len(input_bit))  # max条形图的横坐标
# index_min = index_max + bar_width  # min条形图的横坐标
# # plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
# plt.bar(index_max + bar_width / 2, height=W_time, width=bar_width, color=['#374C6D', '#374C6D', '#374C6D', '#374C6D', '#a3abbd'])
# # plt.bar(index_min, height=W_neuron, width=bar_width, color='#a3abbd', label='Neuromorphic circuits')
# # plt.legend(loc='upper left', borderaxespad=0.25,fontsize=8, ncol=2)
# # plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), borderaxespad=0., fontsize=8, ncol=2)
# plt.xticks(index_max + bar_width / 2, input_bit,fontproperties='Times New Roman', fontsize=9)
# plt.yticks(fontproperties='Times New Roman', fontsize=8)
# # plt.ylabel("Energy consumption($kT$)",fontproperties='Times New Roman', fontsize=14)
#
# plt.plot(x0, y0, color='orange', linewidth=1.5)
# # plt.plot(x1, y1, color='orange', linewidth=1.5)
# plt.annotate('', xy=(0.75, 53020), xytext=(0.75, 680000), arrowprops={'arrowstyle': '<->', 'color': 'orange'})
# # plt.text(0.33, 400, '99.21%', color='#c00000', fontproperties='Times New Roman', fontsize=9)
#
# plt.plot(x2, y2, color='orange', linewidth=1.5)
# # plt.plot(x3, y3, color='orange', linewidth=1.5)
# plt.annotate('', xy=(1.75, 53020), xytext=(1.75, 680000), arrowprops={'arrowstyle': '<->', 'color': 'orange'})
# # plt.text(1.33, 600, '97.97%', color='#c00000', fontproperties='Times New Roman', fontsize=9)
#
# plt.plot(x4, y4, color='orange', linewidth=1.5)
# # plt.plot(x5, y5, color='orange', linewidth=1.5)
# plt.annotate('', xy=(2.75, 53020), xytext=(2.75, 470000), arrowprops={'arrowstyle': '<->', 'color': 'orange'})
# # plt.text(2.33, 550, '91.52%', color='#c00000', fontproperties='Times New Roman', fontsize=9)
#
# plt.plot(x6, y6, color='orange', linewidth=1.5)
# # plt.plot(x7, y7, color='orange', linewidth=1.5)
# plt.annotate('', xy=(3.75, 53020), xytext=(3.75, 940000), arrowprops={'arrowstyle': '<->', 'color': 'orange'})
# # plt.text(3.33, 1200, '98.99%', color='#c00000', fontproperties='Times New Roman', fontsize=9)
#
# plt.plot(x8, y8, linestyle='--', color='orange', linewidth=1.5)
# plt.plot(x9, y9, color='orange', linewidth=1.5)
#
# plt.yscale('log')
#
# plt.tight_layout()
# plt.savefig('logic_时延对比.jpg', format='jpg', bbox_inches='tight')

##########################################logic绘图_时延对比############################################