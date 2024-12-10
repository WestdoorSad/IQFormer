import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd
sns.set(style="white",font='Times New Roman',font_scale=1.0)
# def plot_tsne(features, labels,classes):
#     '''
#     features:(N*m) N*m大小特征，其中N代表有N个数据，每个数据m维
#     label:(N) 有N个标签
#     '''
#     X_tsne = TSNE(n_components=2, random_state=33).fit_transform(features)
#     fig = plt.figure(figsize=(10, 10))
#     ax = Axes3D(fig)
#     fig.add_axes(ax)
#     ax.set_facecolor('w')
#     color = labels
#     b = []
#     for item in classes:
#         b.append(item)
    # scatter = ax.scatter(X_tsne[:, 0], X_tsne[:, 1], c=color, cmap='rainbow')
#     a, _ = scatter.legend_elements()
#     ax.legend(a,b, title="Classes")
#     sns.set_style("whitegrid")
#     plt.savefig('./tsne_A.svg', dpi=450)
#     plt.show()
def plot_tsne(features, labels,dataset,snr,classes):
    '''
    features:(N*m) N*m大小特征，其中N代表有N个数据，每个数据m维
    label:(N) 有N个标签
    '''
    sns.set(style="white")
    X_tsne = TSNE(n_components=2, random_state=33).fit_transform(features)
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot()
    color = labels
    if dataset == '2016.10b':
        color_map = ['#008955', '#5E78B7', '#3A84B7','#68C3E7', '#00CFFF', '#D15D70', '#FFCA99','#F39530','#84F9BD','#AC4978']
    # ['8PSK', 'BPSK', 'CPFSK', 'GFSK', 'PAM4', 'QAM16', 'QAM64', 'QPSK', 'AM-DSB', 'WBFM']
    else:
        color_map = ['#008955', '#5E78B7', '#3A84B7','#68C3E7', '#00CFFF', '#D15D70', '#FFCA99','#F39530','#84F9BD','#B1EA15','#AC4978']
    # b = []
    # for item in classes:
    #     b.append(item)
    # for item in classes:
    #     b.append(item)
    # scatter=plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels,cmap='rainbow')
    df = pd.DataFrame()
    df["y"] = labels
    df["comp1"] = X_tsne[:, 0] 
    df["comp2"] = X_tsne[:, 1]

    sns.scatterplot(x= df.comp1.tolist(), y= df.comp2.tolist(),hue=df.y.tolist(),
                    palette=sns.color_palette(color_map,len(color_map)),edgecolor="none",
                    data=df)
    handles, labels = ax.get_legend_handles_labels()    
    ax.legend(handles, classes,fontsize=12,)
    # a, _ = scatter.legend_elements()
    # plt.legend(a,b, title="Classes")
    plt.title(f'Visualization of t-SNE method at SNR = {snr}dB',fontsize=25)
    ax.set_xticks([])  # 关闭x轴上的刻度
    ax.set_yticks([])  # 关闭y轴上的刻度
    plt.savefig(f'./tsne_{dataset}_{snr}.pdf', dpi=450)
    plt.close()

if __name__ == '__main__':
    digits = datasets.load_digits(n_class=11)
    features, labels = digits.data, digits.target
    print(features.shape)
    print(labels.shape)
    plot_tsne(features, labels)
