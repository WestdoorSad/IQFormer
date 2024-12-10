import numpy as np
import torch
from tqdm import tqdm

'''
In this file, we implement the 'rotation' data augmentation operations 
for modulation recognition mentioned in paper:
    "Data Augmentation for Deep Learning-based Radio Modulation Classification"
    https://arxiv.org/pdf/1912.03026.pdf
'''


def rotation_2d(x):
    x_aug = np.empty(x.shape)
    angs = [0, 90, 180, 270]
    ang = np.random.choice(angs, 1, replace=False)
    if ang == 0:
        x_aug = x
    elif ang == 90:
        x_aug[0, :] = -x[1, :]
        x_aug[1, :] = x[0, :]
    elif ang == 180:
        x_aug = -x
    elif ang == 270:
        x_aug[0, :] = x[1, :]
        x_aug[1, :] = -x[0, :]
    else:
        print("Wrong input for rotation!")
    return x_aug


def Permuting(x):
    x = np.array_split(x, 8, axis=1)
    x = np.random.permutation(x)
    x = np.concatenate(x, axis=1)
    return x


def Reverse(x):
    if np.random.choice([0, 1], 1, replace=False) == 1:
        return np.flip(x, axis=1).copy()
    else:
        return x


def Flip(x):
    if np.random.choice([0, 1], 1, replace=False) == 1:
        x[0] = -x[0]
        return x
    else:
        return x


def normalize_data(X_train):
    # Normalize the data
    m = np.max(np.absolute(X_train))
    X_train = X_train / m
    return X_train


def compute_htcs(IQ):
    """
    Compute Hyperbolic-Tangent Cyclic Spectrum of IQ signal

    Args:
        IQ (np.ndarray): Input IQ signal tensor with shape (batch, 2, 128)

    Returns:
        np.ndarray: Hyperbolic-Tangent Cyclic Spectrum tensor with shape (batch, 128, 128)
    """
    # 将IQ信号转换为复数形式
    IQ = IQ.detach().cpu().numpy()
    comp_signal = IQ[:, 0] + 1j * IQ[:, 1]

    # 计算FFT
    fft_signal = np.fft.fft(comp_signal)

    # 计算HTCS
    htcs = np.zeros((IQ.shape[0], IQ.shape[2], IQ.shape[2]), dtype=np.float32)
    for k in range(IQ.shape[2]):
        for l in range(IQ.shape[2]):
            if k < l:
                htcs[:, k, l] = np.tanh(np.real(fft_signal[:, k]) * np.imag(fft_signal[:, l]))
                htcs[:, l, k] = htcs[:, k, l]

    htcs = torch.FloatTensor(htcs)

    return htcs


# print(Flip(np.array([[1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 4, 5, 6, 7, 8]])))

def jitter(x, sigma=0.03):
    # https://arxiv.org/pdf/1706.00527.pdf
    return x + np.random.normal(loc=0., scale=sigma, size=x.shape)

def scaling(x, sigma=0.1):
    # https://arxiv.org/pdf/1706.00527.pdf
    factor = np.random.normal(loc=1., scale=sigma, size=(x.shape[0],x.shape[2]))
    return np.multiply(x, factor[:,np.newaxis,:])

def rotation(x):
    flip = np.random.choice([-1, 1], size=(x.shape[0],x.shape[2]))
    rotate_axis = np.arange(x.shape[2])
    np.random.shuffle(rotate_axis)    
    return flip[:,np.newaxis,:] * x[:,:,rotate_axis]

def permutation(x, max_segments=5, seg_mode="equal"):
    orig_steps = np.arange(x.shape[1])
    
    num_segs = np.random.randint(1, max_segments, size=(x.shape[0]))
    
    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        if num_segs[i] > 1:
            if seg_mode == "random":
                split_points = np.random.choice(x.shape[1]-2, num_segs[i]-1, replace=False)
                split_points.sort()
                splits = np.split(orig_steps, split_points)
            else:
                splits = np.array_split(orig_steps, num_segs[i])
            warp = np.concatenate(np.random.permutation(splits)).ravel()
            ret[i] = pat[warp]
        else:
            ret[i] = pat
    return ret

def magnitude_warp(x, sigma=0.2, knot=4):
    from scipy.interpolate import CubicSpline
    orig_steps = np.arange(x.shape[1])
    
    random_warps = np.random.normal(loc=1.0, scale=sigma, size=(x.shape[0], knot+2, x.shape[2]))
    warp_steps = (np.ones((x.shape[2],1))*(np.linspace(0, x.shape[1]-1., num=knot+2))).T
    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        warper = np.array([CubicSpline(warp_steps[:,dim], random_warps[i,:,dim])(orig_steps) for dim in range(x.shape[2])]).T
        ret[i] = pat * warper

    return ret

def time_warp(x, sigma=0.2, knot=4):
    from scipy.interpolate import CubicSpline
    orig_steps = np.arange(x.shape[1])
    
    random_warps = np.random.normal(loc=1.0, scale=sigma, size=(x.shape[0], knot+2, x.shape[2]))
    warp_steps = (np.ones((x.shape[2],1))*(np.linspace(0, x.shape[1]-1., num=knot+2))).T
    
    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        for dim in range(x.shape[2]):
            time_warp = CubicSpline(warp_steps[:,dim], warp_steps[:,dim] * random_warps[i,:,dim])(orig_steps)
            scale = (x.shape[1]-1)/time_warp[-1]
            ret[i,:,dim] = np.interp(orig_steps, np.clip(scale*time_warp, 0, x.shape[1]-1), pat[:,dim]).T
    return ret

def window_slice(x, reduce_ratio=0.9):
    # https://halshs.archives-ouvertes.fr/halshs-01357973/document
    target_len = np.ceil(reduce_ratio*x.shape[1]).astype(int)
    if target_len >= x.shape[1]:
        return x
    starts = np.random.randint(low=0, high=x.shape[1]-target_len, size=(x.shape[0])).astype(int)
    ends = (target_len + starts).astype(int)
    
    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        for dim in range(x.shape[2]):
            ret[i,:,dim] = np.interp(np.linspace(0, target_len, num=x.shape[1]), np.arange(target_len), pat[starts[i]:ends[i],dim]).T
    return ret

def window_warp(x, window_ratio=0.1, scales=[0.5, 2.]):
    # https://halshs.archives-ouvertes.fr/halshs-01357973/document
    warp_scales = np.random.choice(scales, x.shape[0])
    warp_size = np.ceil(window_ratio*x.shape[1]).astype(int)
    window_steps = np.arange(warp_size)
        
    window_starts = np.random.randint(low=1, high=x.shape[1]-warp_size-1, size=(x.shape[0])).astype(int)
    window_ends = (window_starts + warp_size).astype(int)
            
    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        for dim in range(x.shape[2]):
            start_seg = pat[:window_starts[i],dim]
            window_seg = np.interp(np.linspace(0, warp_size-1, num=int(warp_size*warp_scales[i])), window_steps, pat[window_starts[i]:window_ends[i],dim])
            end_seg = pat[window_ends[i]:,dim]
            warped = np.concatenate((start_seg, window_seg, end_seg))                
            ret[i,:,dim] = np.interp(np.arange(x.shape[1]), np.linspace(0, x.shape[1]-1., num=warped.size), warped).T
    return ret