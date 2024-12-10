# coding=utf-8
import heapq
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import h5py
from mpl_toolkits.mplot3d import Axes3D
from scipy.signal import windows,stft
plt.rcParams['font.sans-serif'] = ['Times New Roman']  # 如果要显示中文字体,则在此处设为：SimHei
plt.rcParams['axes.unicode_minus'] = False  # 显示负号
import math
from sklearn.preprocessing import MinMaxScaler
import time
import librosa.core as lc

sns.set_theme(font='Times New Roman', font_scale=2.0)

def plot_line(A, B, C, D, label):
    x = np.array(range(0, max(len(A), len(B), len(C), len(D)), 5))

    # label在图示(legend)中显示。若为数学公式,则最好在字符串前后添加"$"符号
    # color：b:blue、g:green、r:red、c:cyan、m:magenta、y:yellow、k:black、w:white、、、
    # 线型：-  --   -.  :    ,
    # marker：.  ,   o   v    <    *    +    1
    plt.figure(figsize=(7, 5))
    plt.grid(linestyle="-")  # 设置背景网格线为虚线
    ax = plt.gca()
    ax.spines['top'].set_visible(False)  # 去掉上边框
    ax.spines['right'].set_visible(False)  # 去掉右边框

    plt.plot(A, color="cornflowerblue", label=f"Original Model", linewidth=1.5)
    plt.plot(B, color="red", label=f"Without Reverse", linewidth=1.5)
    plt.plot(C, color="olivedrab", label=f"Without Rotation", linewidth=1.5)
    plt.plot(D, color="darkorange", label=f"Without Rotation&Reverse", linewidth=1.5)
    plt.title('Train Accuracy', fontsize=15)
    group_labels = range(0, max(len(A), len(B), len(C), len(D)), 5)  # x轴刻度的标识
    plt.xticks(x, group_labels, fontsize=15, fontweight='bold')  # 默认字体大小为10
    plt.yticks(fontsize=15, fontweight='bold')
    plt.xlabel("Epoch", fontsize=15, fontweight='bold')
    if label == 'loss':
        plt.ylabel("Loss", fontsize=15, fontweight='bold')
        plt.xlim(0, 85)  # 设置x轴的范围
        plt.ylim(1.0, 1.5)
    else:
        plt.ylabel("Accuracy", fontsize=15, fontweight='bold')
        plt.xlim(0, 85)  # 设置x轴的范围
        plt.ylim(0.40, 0.7)

    # plt.legend()          #显示各曲线的图例
    plt.legend(loc=0, numpoints=1)
    leg = plt.gca().get_legend()
    ltext = leg.get_texts()
    plt.setp(ltext, fontsize=12, fontweight='bold')  # 设置图例字体的大小和粗细

    plt.savefig(f'./Train {label}.svg', format='svg')  # 建议保存为svg格式,再用inkscape转为矢量图emf后插入word中
    plt.show()


def plot_modulations():
    data = pd.read_pickle('../dataset/RML2016.10a_dict.pkl')
    vis = []
    for item in data.items():
        (label, SNR), samples = item
        if SNR < 18:
            continue
        vis.append([label, samples[25]])
    plt.subplot(341)
    plt.plot(vis[0][1][0], color="cornflowerblue")
    plt.plot(vis[0][1][1], color="lightcoral")
    plt.title(vis[0][0])
    plt.xticks([])
    plt.yticks([])
    plt.subplot(342)
    plt.plot(vis[1][1][0], color="cornflowerblue")
    plt.plot(vis[1][1][1], color="lightcoral")
    plt.title(vis[1][0])
    plt.xticks([])
    plt.yticks([])
    plt.subplot(343)
    plt.plot(vis[2][1][0], color="cornflowerblue")
    plt.plot(vis[2][1][1], color="lightcoral")
    plt.title(vis[2][0])
    plt.xticks([])
    plt.yticks([])
    plt.subplot(344)
    plt.plot(vis[3][1][0], color="cornflowerblue")
    plt.plot(vis[3][1][1], color="lightcoral")
    plt.title(vis[3][0])
    plt.xticks([])
    plt.yticks([])
    plt.subplot(345)
    plt.plot(vis[4][1][0], color="cornflowerblue")
    plt.plot(vis[4][1][1], color="lightcoral")
    plt.title(vis[4][0])
    plt.xticks([])
    plt.yticks([])
    plt.subplot(346)
    plt.plot(vis[5][1][0], color="cornflowerblue")
    plt.plot(vis[5][1][1], color="lightcoral")
    plt.title(vis[5][0])
    plt.xticks([])
    plt.yticks([])
    plt.subplot(347)
    plt.plot(vis[6][1][0], color="cornflowerblue")
    plt.plot(vis[6][1][1], color="lightcoral")
    plt.title(vis[6][0])
    plt.xticks([])
    plt.yticks([])
    plt.subplot(348)
    plt.plot(vis[7][1][0], color="cornflowerblue")
    plt.plot(vis[7][1][1], color="lightcoral")
    plt.title(vis[7][0])
    plt.xticks([])
    plt.yticks([])
    plt.subplot(349)
    plt.plot(vis[8][1][0], color="cornflowerblue")
    plt.plot(vis[8][1][1], color="lightcoral")
    plt.title(vis[8][0])
    plt.xticks([])
    plt.yticks([])
    plt.subplot(3, 4, 10)
    plt.plot(vis[9][1][0], color="cornflowerblue")
    plt.plot(vis[9][1][1], color="lightcoral")
    plt.title(vis[9][0])
    plt.xticks([])
    plt.yticks([])
    plt.subplot(3, 4, 11)
    plt.plot(vis[10][1][0], color="cornflowerblue")
    plt.plot(vis[10][1][1], color="lightcoral")
    plt.title(vis[10][0])
    plt.xticks([])
    plt.yticks([])
    plt.savefig(f'visualize_of_modulations.svg', format='svg', dpi=450)
    plt.show()

def center_f(data):
    sum_f = []
    for i in range(data.shape[0]):
        f_sum = sum(data[i])
        print(f_sum)
        sum_f.append(f_sum)
    max_number = heapq.nlargest(2,sum_f)
    max_idx = []
    for t in max_number:    
        index = sum_f.index(t)
        max_idx.append(index)
        sum_f[index] = 0
    print(max_idx)
    idx = np.array(max_idx).mean()
    return int(idx)

def rotation_2d(x):
    x_aug1 = np.empty(x.shape)
    x_aug2 = np.empty(x.shape)
    x_aug3 = np.empty(x.shape)    
    x_aug1[0, :] = -x[1, :]
    x_aug1[1, :] = x[0, :]
    x_aug2 = -x
    x_aug3[0, :] = x[1, :]
    x_aug3[1, :] = -x[0, :]
    return x_aug1,x_aug2,x_aug3
classes = ['BPSK', 'QPSK', '8PSK', '16PSK', '32PSK', '64PSK', '4QAM', '8QAM', '16QAM', '32QAM', 
                   '64QAM', '128QAM', '256QAM', '2FSK', '4FSK', '8FSK', '16FSK', '4PAM', '8PAM', '16PAM', 'AM-DSB', 
                   'AM-DSB-SC', 'AM-USB', 'AM-LSB', 'FM', 'PM']
with h5py.File('./dataset/HisarMod2019test.h5') as h5file:
    test = h5file['samples'][:]
    test_label = h5file['labels'][:]
    SNR_te = h5file['snr'][:]
    h5file.close()
snr_idx = np.where(SNR_te==16)[0]
print('test_index_lenth:',len(snr_idx))
test = test[snr_idx]
test_label = test_label[snr_idx]
SNR_te = SNR_te[snr_idx]
vis = []
for type in range(len(classes)):    
    snr_idx = np.where(test_label==type)[0]
    vis.append([classes[type],test[snr_idx[0]]])

n_fft=64
hop_length=2
for x in vis:
    title = x[0]
    IQ = x[1]
    z = np.arange(128)
    fig = plt.figure('image1',frameon=False)
    plt.plot(IQ[0,:],color='cornflowerblue')
    plt.plot(IQ[1,:], color='lightcoral')
    plt.axis('off')
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace=0,wspace=0)
    plt.margins(0,0)
    plt.tight_layout() 
    plt.title(f'{x[0]}')
    plt.savefig(f'./dataset/IQ.png')
    plt.show()
    fig = plt.figure('image2',frameon=False)
    f,t,data01 = stft(IQ[1,:],1.0,'blackman',61,60,128)

    
    # data01 = np.array(lc.stft(IQ[0,:], n_fft=n_fft, hop_length=hop_length, win_length=25, window=windows.blackman(25, sym=False)))
    # f,t,data01 = stft(IQ[0,:],sr,'blackman',13,12,64)
    print(data01.shape)
    plt.pcolormesh(t, f, abs(data01))
    plt.axis('off')
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace=0,wspace=0)
    plt.margins(0,0)
    plt.tight_layout() 
    plt.title(f'{x[0]}')
    plt.savefig(f'./dataset/STFT.png')
    plt.show()
    # fig = plt.figure()
    # ax = Axes3D(fig)
    # fig.add_axes(ax)
    # ax.set_facecolor('w')
    # scatter = ax.scatter(z,IQ[0,:], IQ[1,:])
    # ax.set_title(title)
    # ax.set_xlabel('I',fontsize=14)
    # ax.set_ylabel('Q',fontsize=14)
    # ax.set_zlabel('timestep',fontsize=14)
    # plt.show()
    # IQ_STFT = np.empty(shape=(128), dtype=np.complex128)
    # # f, t, zxx = stft(IQ_STFT, fs=200000, noverlap=23, nfft=128,nperseg=24)
    # data01 = np.array(lc.stft(IQ[0,:], n_fft=n_fft, hop_length=hop_length, win_length=32, window=windows.blackman(32, sym=True)))
    # plt.title(f'{x[0]}')
    # plt.pcolormesh(np.array(range(int(len(IQ_STFT)/hop_length+1)))/sr, sr*np.array(range(int(1+n_fft/2)))/(n_fft/2), np.abs(data01))
    # plt.ylabel('Frequency [Hz]')
    # plt.xlabel('Time [sec]')
    # plt.tight_layout() 
    # plt.show()
    # tool = MinMaxScaler(feature_range=(0, 1))
    # data01 = tool.fit_transform(np.abs(data01)) 
    # for i in range(data01.shape[0]):
    #     for j in range(data01.shape[1]):
    #         if data01[i][j] < 0.2:
    #             data01[i][j] = 0
    # fig = plt.figure('image',frameon=False)
    # print(data01.shape)
    # # fig.set_size_inches(256/100, 256/100)
    # plt.pcolormesh(np.array(range(int(len(IQ_STFT)/hop_length+1)))/sr, sr*np.array(range(int(1+n_fft/2)))/(n_fft/2), np.abs(data01))
    # plt.show()