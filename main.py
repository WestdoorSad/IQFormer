import argparse
import os
import h5py
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from torch.optim.lr_scheduler import ReduceLROnPlateau
from dataset.dataset import RMLgeneral, RMLval, RMLtest
import numpy as np
from torch.utils.data import DataLoader
from model.IQFormer import IQFormer
import torch
from torch import nn
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
from utils.plot_tSNE import plot_tsne
from utils.traintest import train_epoch, test_epoch, val_epoch
import time
# plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
# plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


def plot_confusion_matrix(cm, database,SNR , labels=[]):
    plt.figure(figsize=(10, 10))
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='.2f', xticklabels=labels, yticklabels=labels, cbar=False,
                square=True,
                annot_kws={"fontsize": 20})
    plt.title(database+' SNR='+str(SNR)+'dB')
    plt.xticks(fontsize=20, rotation=45)
    plt.yticks(fontsize=20, rotation=0)
    plt.tight_layout()
    plt.savefig(f'{database}_{SNR}.pdf', bbox_inches='tight', dpi=450)
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('RML SMY model')
    # Dataset
    parser.add_argument('--database_path', type=str, default="./dataset")
    parser.add_argument('--database_choose', type=str, default="2016.10a")
    # Hyperparameters
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--minSNR', type=int, default=-20) # 2016 -20-18 2018 -20-30
    parser.add_argument('--maxSNR', type=int, default=18)
    parser.add_argument('--test_size', type=float, default=0.2)
    parser.add_argument('--eval_batch_size', type=int, default=400)
    parser.add_argument('--num_epochs', type=int, default=60)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--loss', type=str, default='CCE', help='Cross Entropy Loss')

    # model
    parser.add_argument('--seed', type=int, default=1234,
                        help='random seed (default: 1234)')
    parser.add_argument('--model_path', type=str,
                        default=None, help='Model checkpoint')
    parser.add_argument('--comment', type=str, default='IQFormer',
                        help='Comment to describe the saved model')
    if not os.path.exists('save_models'):
        os.mkdir('save_models')
    args = parser.parse_args()

    if not os.path.exists('save_models'):
        os.mkdir('save_models')
    args = parser.parse_args()

    # define model saving path
    model_tag = 'model_{}_{}_{}_{}'.format(args.database_choose, args.num_epochs, args.batch_size, args.lr)
    if args.comment:
        model_tag = model_tag + '_{}'.format(args.comment)
    model_save_path = os.path.join('save_models', model_tag)

    # set model save directory
    if not os.path.exists(model_save_path):
        os.mkdir(model_save_path)

    # GPU device
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print('Device: {}'.format(device))

    # Load dataset
    train_dataset = [[],[],[]]
    val_dataset = [[],[],[]]
    test_dataset = [[],[],[]]
    if args.database_choose == '2019':
        classes = ['BPSK', 'QPSK', '8PSK', '16PSK', '32PSK', '64PSK', '4QAM', '8QAM', '16QAM', '32QAM', 
                   '64QAM', '128QAM', '256QAM', '2FSK', '4FSK', '8FSK', '16FSK', '4PAM', '8PAM', '16PAM', 'AM-DSB', 
                   'AM-DSB-SC', 'AM-USB', 'AM-LSB', 'FM', 'PM']
        with h5py.File(os.path.join(args.database_path, 'HisarMod2019train.h5')) as h5file:
            train = h5file['samples'][:]
            train_label = h5file['labels'][:]
            SNR_tr = h5file['snr'][:]
            h5file.close()
        snr_idx = np.where((SNR_tr>= args.minSNR) & (SNR_tr<= args.maxSNR))[0]
        print(train.shape)
        print('train_index_lenth:',len(snr_idx))
        train = train[snr_idx]
        train_label = train_label[snr_idx]
        SNR_tr = SNR_tr[snr_idx]
        train, val, train_label, val_label, SNR_tr, SNR_va = train_test_split(train, train_label, SNR_tr, test_size=args.test_size,
                                                                            random_state=233,
                                                                            stratify=list(zip(train_label,SNR_tr)))
        with h5py.File(os.path.join(args.database_path, 'HisarMod2019test.h5')) as h5file:
            test = h5file['samples'][:]
            test_label = h5file['labels'][:]
            SNR_te = h5file['snr'][:]
            h5file.close()
        snr_idx = np.where((SNR_te>= args.minSNR) & (SNR_te<= args.maxSNR))[0]
        print('test_index_lenth:',len(snr_idx))
        test = test[snr_idx]
        test_label = test_label[snr_idx]
        SNR_te = SNR_te[snr_idx]
        train_dataset = RMLgeneral(train,train_label,SNR_tr)
        val_dataset = RMLval(val,val_label,SNR_va)
        test_dataset = RMLtest(test,test_label,SNR_te)
    else: 
        if args.database_choose[-1] == 'a':
            data = pd.read_pickle(os.path.join(args.database_path, 'RML2016.10a_dict.pkl'))
            classes = ['8PSK', 'BPSK', 'CPFSK', 'GFSK', 'PAM4', 'QAM16', 'QAM64', 'QPSK', 'AM-DSB', 'AM-SSB', 'WBFM']
        else:
            classes = ['8PSK', 'BPSK', 'CPFSK', 'GFSK', 'PAM4', 'QAM16', 'QAM64', 'QPSK', 'AM-DSB',
                    'WBFM']
            data = pd.read_pickle(os.path.join(args.database_path, 'RML2016.10b.dat'))
        train_dataset = [[],[],[]]
        val_dataset = [[],[],[]]
        test_dataset = [[],[],[]]
        for item in data.items():
            (label, SNR), samples = item
            if SNR < args.minSNR or SNR > args.maxSNR or label not in classes:
                continue
            labels = np.full(len(samples), classes.index(label))
            SNR = np.full(len(samples), SNR)
            X, x, Y, y, SNR_tr, SNR_te = train_test_split(samples, labels, SNR, test_size=args.test_size,
                                                        random_state=233,
                                                        stratify=labels)
            train, val, train_label, val_label, SNR_tr, SNR_va = train_test_split(X, Y, SNR_tr, test_size=0.25,
                                                                                random_state=233,
                                                                                stratify=Y)
            train_dataset[0].extend(train)
            train_dataset[1].extend(train_label)
            train_dataset[2].extend(SNR_tr)
            val_dataset[0].extend(val)
            val_dataset[1].extend(val_label)
            val_dataset[2].extend(SNR_va)
            test_dataset[0].extend(x)
            test_dataset[1].extend(y)
            test_dataset[2].extend(SNR_te)
        train_dataset = RMLgeneral(np.array(train_dataset[0]),np.array(train_dataset[1]),np.array(train_dataset[2]))
        val_dataset = RMLtest(np.array(val_dataset[0]),np.array(val_dataset[1]),np.array(val_dataset[2]))
        test_dataset = RMLtest(np.array(test_dataset[0]),np.array(test_dataset[1]),np.array(test_dataset[2]))
    print(f'train_size:{len(train_dataset)}\tval_size:{len(val_dataset)}\t')
    # Training Dataloader
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
    # Testing Dataloader
    val_loader = DataLoader(
        val_dataset, batch_size=args.eval_batch_size, shuffle=False, drop_last=False)
    del train_dataset
    del val_dataset
    test_loader = DataLoader(test_dataset, batch_size=args.eval_batch_size, shuffle=False, drop_last=False)
    if args.database_choose == '2016.10b':
        num_classes = 10
    elif args.database_choose == '2019':
        num_classes = 26
    else:
        num_classes = 11
    if args.database_choose in ['2016.10a','2016.10b']:
        model = IQFormer([2,3,2], embed_dims=[64,64,64],
                mlp_ratios=4,
                act_layer=nn.GELU,
                num_classes=num_classes,
                down_patch_size=3, down_stride=2, down_pad=1,
                drop_rate=0.2, drop_path_rate=0.,
                use_layer_scale=False, layer_scale_init_value=1e-5,
                fork_feat=False,
                vit_num=1,)
    else:
        model = IQFormer([3,3,3], embed_dims=[64,64,64],
            mlp_ratios=4,
            act_layer=nn.GELU,
            num_classes=num_classes,
            down_patch_size=3, down_stride=2, down_pad=1,
            drop_rate=0.2, drop_path_rate=0.2,
            use_layer_scale=False, layer_scale_init_value=1e-5,
            fork_feat=False,
            vit_num=1,)
    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')
    # criterion
    criterion = nn.CrossEntropyLoss()
    # AdamW optimizer
    optimizer1 = torch.optim.AdamW(model.parameters(),  lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer1, 'min', factor=0.5, patience=3, verbose=True,min_lr=5e-5)

    if args.model_path:
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print('Model loaded : {}'.format(args.model_path))

    # Training and testing
    epochs_without_improvement = 0
    patience = 10
    num_epochs = args.num_epochs
    writer = SummaryWriter('logs/{}'.format(model_tag))
    for epoch in range(num_epochs):
        train_loss, train_ACC, train_true, train_pred = train_epoch(epoch, train_loader, model, args.minSNR,
                                                                    args.maxSNR,optimizer1, criterion, device)
                                                                     
        writer.add_scalar('train_loss', train_loss, epoch)
        val_loss, val_ACC, val_true, val_pred, model_v = val_epoch(epoch, val_loader, model, args.minSNR,
                                                                   args.maxSNR,
                                                                   scheduler, criterion, device)
        writer.add_scalar('val_loss', val_loss, epoch)
        if epoch == 0:
            torch.save(model.state_dict(), os.path.join(model_save_path, 'weight.pt'))
            max_acc = val_ACC['Avg']

        else:
            if max_acc < val_ACC['Avg']:
                torch.save(model.state_dict(), os.path.join(model_save_path, 'weight.pt'))
                avg = val_ACC['Avg']
                print(f'max_acc:{max_acc}=====>{avg}')
                max_acc = val_ACC['Avg']
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                print(f'epochs_without_improvement:{epochs_without_improvement}/{patience}')
        tracc = pd.DataFrame.from_dict(train_ACC, orient='index', columns=['0']).reset_index(names='SNR')
        vaacc = pd.DataFrame.from_dict(val_ACC, orient='index', columns=['0']).reset_index(names='SNR')
        if epoch == 0:
            tracc.to_csv(f'logs/{model_tag}/Train_Epoch.csv', index=False)
            vaacc.to_csv(f'logs/{model_tag}/Val_Epoch.csv', index=False)
        else:
            tracc = pd.Series(tracc['0'])
            vaacc = pd.Series(vaacc['0'])
            tr = pd.read_csv(f'logs/{model_tag}/Train_Epoch.csv')
            tr.insert(tr.shape[1], f'{epoch}', tracc)
            va = pd.read_csv(f'logs/{model_tag}/Val_Epoch.csv')
            va.insert(va.shape[1], f'{epoch}', vaacc)
            tr.to_csv(f'logs/{model_tag}/Train_Epoch.csv', index=False)
            va.to_csv(f'logs/{model_tag}/Val_Epoch.csv', index=False)

        train_CM = confusion_matrix(train_true, train_pred)
        traincm = train_CM.astype('float') / train_CM.sum(axis=1)[:, np.newaxis]  # 归一化
        traincm = np.around(traincm, decimals=2)
        val_CM = confusion_matrix(val_true, val_pred)
        valcm = val_CM.astype('float') / val_CM.sum(axis=1)[:, np.newaxis]  # 归一化
        valcm = np.around(valcm, decimals=2)

        #   plotCM
        plot_confusion_matrix(traincm, args.database_choose,'all', labels=classes)
        plot_confusion_matrix(valcm, args.database_choose,'all', labels=classes)
        if epochs_without_improvement == patience:
            print(f'Early stopping at epoch {epoch}')
            break
    del train_loader
    del val_loader
    model.load_state_dict(torch.load(os.path.join(model_save_path, 'weight.pt')))
    start = time.time()
    test_true, test_pred, test_SNR = test_epoch(0, test_loader, model, device)
    end = time.time()
    used = end - start
    print('Avg_test_time:', used / len(test_pred))
    pred = torch.stack(test_pred).cpu().data.numpy().argmax(1).tolist()
    true = torch.stack(test_true).cpu().data.numpy().tolist()
    test_SNR = [int(i) for i in test_SNR]
    SNR = dict([(key, 0) for key in range(args.minSNR, args.maxSNR + 1, 2)])
    SNR_true = dict([(key, 0) for key in range(args.minSNR, args.maxSNR + 1, 2)])
    for slice in range(len(pred)):
        if (type(test_SNR[slice])).__name__ == 'list':
            test_SNR[slice] = test_SNR[slice][0]
        if pred[slice] == true[slice]:  
            SNR[test_SNR[slice]] = SNR.get(test_SNR[slice]) + 1
            SNR_true[test_SNR[slice]] = SNR_true.get(test_SNR[slice]) + 1
        else:
            SNR[test_SNR[slice]] = SNR.get(test_SNR[slice]) + 1
    avg_true = 0
    avg_all = 0
    for key in range(args.minSNR, args.maxSNR + 1, 2):
        avg_all += SNR[key]
        avg_true += SNR_true[key]
        SNR[key] = SNR_true[key] / float(SNR[key])
    SNR['Avg'] = avg_true / float(avg_all)
    Avg = SNR['Avg']
    print(f'test_acc={Avg}')
    # Save test accuracy
    testacc = pd.DataFrame.from_dict(SNR, orient='index', columns=['0']).reset_index(names='SNR')
    testacc.to_csv(f'logs/{model_tag}/Test_ACC.csv', index=False)
    mod_dic = {}
    for snr in range(args.minSNR,args.maxSNR+1,2):
        SNR_cm = [i for i in zip(test_SNR, pred, true) if i[0] == snr]
        true_cm = []
        pred_cm = []
        true_cls = np.zeros(num_classes)
        all = np.zeros(num_classes)
        for i in SNR_cm:
            pred_cm.append(i[1])
            true_cm.append(i[2])
            if i[1] == i[2]:
                true_cls[i[1]] = true_cls[i[1]] + 1
                all[i[1]] = all[i[1]] + 1
            else:
                all[i[2]] = all[i[2]] + 1
        Cls_ACC = {cls: x / y for cls, x, y in zip(classes, true_cls, all)}
        mod_dic[snr] = list(Cls_ACC.values())
        modacc = pd.DataFrame.from_dict(mod_dic, orient='index', columns=classes).reset_index(names='SNR')
        modacc.to_csv(f'logs/{model_tag}/Test_mod_SNR.csv', index=False)
        test_CM = confusion_matrix(true_cm, pred_cm)
        testcm = test_CM.astype('float') / test_CM.sum(axis=1)[:, np.newaxis]  # 归一化
        testcm = np.around(testcm, decimals=2)
        plot_confusion_matrix(testcm, args.database_choose,snr, labels=classes)
        SNR_tsne_ = [i for i in
                zip(test_SNR, torch.stack(test_pred).cpu().data.numpy(), torch.stack(test_true).cpu().data.numpy()) if
                i[0] == snr]
        _, pred_0, true_0 = zip(*SNR_tsne_)
        plot_tsne(np.array(list(pred_0)), np.array(list(true_0)),args.database_choose,snr, classes)
