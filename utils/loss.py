import heapq
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import h5py
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from scipy.signal import windows,stft
plt.rcParams['font.sans-serif'] = ['Times New Roman']  # 如果要显示中文字体,则在此处设为：SimHei
plt.rcParams['axes.unicode_minus'] = False  # 显示负号
import math
from sklearn.preprocessing import MinMaxScaler
import time
import librosa.core as lc
def plot_line(A, B):
    x = np.array(range(0, max(len(A), len(B)), 5))

    # label在图示(legend)中显示。若为数学公式,则最好在字符串前后添加"$"符号
    # color：b:blue、g:green、r:red、c:cyan、m:magenta、y:yellow、k:black、w:white、、、
    # 线型：-  --   -.  :    ,
    # marker：.  ,   o   v    <    *    +    1
    plt.figure(figsize=(7, 5))
    plt.grid(linestyle="-")  # 设置背景网格线为虚线
    ax = plt.gca()
    ax.spines['top'].set_visible(False)  # 去掉上边框
    ax.spines['right'].set_visible(False)  # 去掉右边框

    plt.plot(A, color="cornflowerblue", label=f"train", linewidth=1.5)
    plt.plot(B, color="red", label=f"val", linewidth=1.5)
    plt.title('RML2016.10b', fontsize=15)
    group_labels = range(0, max(len(A), len(B)), 5)  # x轴刻度的标识
    plt.xticks(x, group_labels, fontsize=15, fontweight='bold')  # 默认字体大小为10
    plt.yticks(fontsize=15, fontweight='bold')
    plt.xlabel("Epoch", fontsize=15, fontweight='bold')
    plt.ylabel("Loss", fontsize=15, fontweight='bold')
    plt.xlim(0, 55)  # 设置x轴的范围
    # plt.legend()          #显示各曲线的图例
    plt.legend(loc=0, numpoints=1)
    leg = plt.gca().get_legend()
    ltext = leg.get_texts()
    plt.setp(ltext, fontsize=12, fontweight='bold')  # 设置图例字体的大小和粗细

    plt.savefig(f'./b.png')
    plt.show()

x = pd.read_csv('E:/Code/IQFormer/utils/b.csv',header=None)
train = x[0].values
val = x[1].values
plot_line(train,val)