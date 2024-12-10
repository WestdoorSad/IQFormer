from tqdm import tqdm
from tqdm import tqdm
import numpy as np
import torch
import time
import sys


def train_epoch(num_epoch, data_loader, model, min_SNR, max_SNR, optimizer, criterion, device):
    running_loss = 0
    num_total = 0.0
    model.train()
    with tqdm(total=len(data_loader)) as t:
        SNR = dict([(key, 0) for key in range(min_SNR, max_SNR + 1, 2)])
        SNR_true = dict([(key, 0) for key in range(min_SNR, max_SNR + 1, 2)])
        y_true = []
        y_pred = []
        for _, data in enumerate(data_loader):
            batch_x, batch_SNR, batch_y = data
            t.set_description('Epoch %i' % num_epoch)
            batch_size = batch_x.size(0)
            num_total += batch_size

            batch_x = batch_x.to(device)          
            batch_SNR = batch_SNR.numpy().tolist()
            batch_y = batch_y.to(device)

            batch_out = model(batch_x)
            batch_loss = criterion(batch_out, batch_y)
            train_pred = batch_out.cpu().detach().numpy()
            train_pred = train_pred.argmax(1).tolist()
            train_true = batch_y.cpu().detach().numpy().tolist()
            # train_true = train_true.argmax(1).tolist()
            y_true.extend(train_true)
            y_pred.extend(train_pred)

            for slice in range(batch_size):
                if (type(batch_SNR[slice])).__name__ == 'list':
                    batch_SNR[slice] = batch_SNR[slice][0]
                if train_pred[slice] == train_true[slice]:
                    SNR[batch_SNR[slice]] = SNR.get(batch_SNR[slice]) + 1
                    SNR_true[batch_SNR[slice]] = SNR_true.get(batch_SNR[slice]) + 1
                else:
                    SNR[batch_SNR[slice]] = SNR.get(batch_SNR[slice]) + 1

            running_loss += (batch_loss.item() * batch_size)
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            t.set_postfix(loss=running_loss)
            t.update(1)

    running_loss /= num_total
    avg_true = 0
    avg_all = 0
    for key in range(min_SNR, max_SNR + 1, 2):
        avg_all += SNR[key]
        avg_true += SNR_true[key]
        SNR[key] = SNR_true[key] / float(SNR[key])
    SNR['Avg'] = avg_true / float(avg_all)
    Avg = SNR['Avg']
    print(f'Epoch:{num_epoch},train_loss={running_loss},train_acc={Avg}')
    return running_loss, SNR, y_true, y_pred


def val_epoch(num_epoch, data_loader, model, min_SNR, max_SNR, scheduler, criterion, device):
    val_loss = 0.0
    num_total = 0.0
    SNR = dict([(key, 0) for key in range(min_SNR, max_SNR + 1, 2)])
    SNR_true = dict([(key, 0) for key in range(min_SNR, max_SNR + 1, 2)])
    y_pred = []
    y_true = []
    model.eval()

    with torch.no_grad():
        with tqdm(total=len(data_loader)) as t:
            for _, data in enumerate(data_loader):
                batch_x, batch_SNR, batch_y = data
                t.set_description('Epoch %i' % num_epoch)
                batch_size = batch_x.size(0)
                num_total += batch_size
                batch_x = batch_x.to(device)
                batch_SNR = batch_SNR.numpy().tolist()
                batch_y = batch_y.to(device)
                start = time.time()
                batch_out = model(batch_x)
                end = time.time()
                batch_loss = criterion(batch_out, batch_y)
                train_pred = batch_out.cpu().detach().numpy()
                train_pred = train_pred.argmax(1).tolist()
                train_true = batch_y.cpu().detach().numpy().tolist()
                y_true.extend(train_true)
                y_pred.extend(train_pred)

                for slice in range(batch_size):
                    if (type(batch_SNR[slice])).__name__ == 'list':
                        batch_SNR[slice] = batch_SNR[slice][0]
                    if train_pred[slice] == train_true[slice]:
                        SNR[batch_SNR[slice]] = SNR.get(batch_SNR[slice]) + 1
                        SNR_true[batch_SNR[slice]] = SNR_true.get(batch_SNR[slice]) + 1
                    else:
                        SNR[batch_SNR[slice]] = SNR.get(batch_SNR[slice]) + 1

                val_loss += (batch_loss.item() * batch_size)
                t.set_postfix(loss=val_loss)
                t.update(1)

        val_loss /= num_total
        scheduler.step(val_loss)

        avg_true = 0
        avg_all = 0
        for key in range(min_SNR, max_SNR + 1, 2):
            avg_all += SNR[key]
            avg_true += SNR_true[key]
            SNR[key] = SNR_true[key] / float(SNR[key])
        SNR['Avg'] = avg_true / float(avg_all)
        Avg = SNR['Avg']
        print(f'Epoch:{num_epoch},val_loss={val_loss},val_acc={Avg}')
    return val_loss, SNR, y_true, y_pred, model


def test_epoch(num_epoch, data_loader, model, device):
    y_pred = []
    y_true = []
    SNR = []
    model.eval()

    with torch.no_grad():
        for _, data in enumerate(data_loader):
            batch_x, batch_SNR, batch_y = data
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            batch_out = model(batch_x)
            y_true.extend(batch_y)
            y_pred.extend(batch_out)
            SNR.extend(batch_SNR)
    return y_true, y_pred, SNR


if __name__ == '__main__':
    SNR = dict([(key, 0) for key in range(-20, 31, 2)])
    SNR_true = dict([(key, 0) for key in range(-20, 31, 2)])
    x = np.array([2, 3, 4, 3, 5, 3])
    y = torch.tensor([1, 3, 5, 2, 5, 4])
    snr = [-20, -20, -20, 14, 10, 10]
    for i in range(len(x)):
        if x[i] == y[i]:
            SNR[snr[i]] = SNR.get(snr[i]) + 1
            SNR_true[snr[i]] = SNR_true.get(snr[i]) + 1
        else:
            SNR[snr[i]] = SNR.get(snr[i]) + 1
    for key in range(-20, 31, 2):
        if SNR[key] != 0:
            SNR[key] = SNR_true[key] / float(SNR[key])
    print(SNR)
