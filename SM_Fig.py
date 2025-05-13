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

##########################################Fig. S1############################################

file_path = "D:\信息与能量研究方向\论文\基于随机热力学的神经形态电路\神经元.xlsx"
# 使用 usecols 选择列，skiprows 跳过不需要的行
delta_t = pd.read_excel(file_path, usecols='S', skiprows=47, nrows=26, sheet_name='间隔', header=None)  # 从第2行起，读取A列到C列，5行数据
diss_12000 = pd.read_excel(file_path, usecols='T', skiprows=47, nrows=26, sheet_name='间隔', header=None)
diss_14000 = pd.read_excel(file_path, usecols='U', skiprows=47, nrows=26, sheet_name='间隔', header=None)
diss_16000 = pd.read_excel(file_path, usecols='V', skiprows=47, nrows=26, sheet_name='间隔', header=None)
diss_18000 = pd.read_excel(file_path, usecols='W', skiprows=47, nrows=26, sheet_name='间隔', header=None)
diss_20000 = pd.read_excel(file_path, usecols='X', skiprows=47, nrows=26, sheet_name='间隔', header=None)
# print(delta_t.iloc[:, 0]. values)

delta_t = delta_t.iloc[:, 0]. values
diss_12000 = diss_12000.iloc[:, 0]. values
diss_14000 = diss_14000.iloc[:, 0]. values
diss_16000 = diss_16000.iloc[:, 0]. values
diss_18000 = diss_18000.iloc[:, 0]. values
diss_20000 = diss_20000.iloc[:, 0]. values

# plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.figure(1, dpi=300, figsize=(5, 3))
plt.rcParams['xtick.direction'] = 'in'  # 将x周的刻度线方向设置向内
plt.rcParams['ytick.direction'] = 'in'  # 将y轴的刻度方向设置向内

plt.plot(delta_t, diss_12000, label='$\Delta t_{\mathrm{in}}$=12000 $β\hbar$', marker="s", markersize=3, color='#c5d9ff')
plt.plot(delta_t, diss_14000, label='$\Delta t_{\mathrm{in}}$=14000 $β\hbar$', marker="s", markersize=3, color='#a0b3da')
plt.plot(delta_t, diss_16000, label='$\Delta t_{\mathrm{in}}$=16000 $β\hbar$', marker="s", markersize=3, color='#7c8fb4')
plt.plot(delta_t, diss_18000, label='$\Delta t_{\mathrm{in}}$=18000 $β\hbar$', marker="s", markersize=3, color='#596d90')
plt.plot(delta_t, diss_20000, label='$\Delta t_{\mathrm{in}}$=20000 $β\hbar$', marker="s", markersize=3, color='#374c6d')

# plt.subplot(2, 1, 1)
# plt.plot(Vctrl, delta_t, label='$\Delta t_{\mathrm{dev}}$($β\hbar$)', marker="s", markersize=4, color='#374C6D')
# plt.yticks(fontproperties='Times New Roman')
# plt.xticks([])
# plt.tick_params(labelsize=8)
# # plt.gca().invert_xaxis()
# plt.legend(loc='upper left', fontsize=9)
# plt.subplot(2, 1, 2)
# plt.plot(Vctrl, diss, label='$E_{\mathrm{neuron}}$($kT$)', marker="s", markersize=4, color='#a3abbd')
plt.xlabel("Input spike interval $\Delta t{_{\mathrm{in1}}^{\mathrm{in2}}}$ ($β\hbar$)", fontsize=9, fontproperties='Times New Roman')
plt.ylabel("Energy consumption $E_{\mathrm{neuron}}$ ($kT$)", fontsize=9, fontproperties='Times New Roman')
plt.yticks(fontproperties='Times New Roman')
plt.xticks(fontproperties='Times New Roman')
plt.tick_params(labelsize=8)

plt.legend(loc='upper left', prop={'family': 'Times New Roman', 'size': 8})
plt.tight_layout()
plt.savefig('补充图1.svg', format='svg', bbox_inches='tight')


##########################################Fig. S1############################################