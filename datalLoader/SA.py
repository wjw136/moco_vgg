import time
import random
import argparse
import sys

import numpy as np
import numpy.linalg as nl
import matplotlib.pyplot as plt
import librosa
import librosa.display
from scipy import interpolate
from scipy import signal
from scipy.io import wavfile
from scipy.fftpack import dct, idct
from scipy.spatial.distance import pdist, cdist, squareform

import cv2
from PIL import Image
from matplotlib import cm


def corp_margin(img):
    img2 = img.sum(axis=2)
    (row, col) = img2.shape
    row_top = 0
    raw_down = 0
    col_top = 0
    col_down = 0
    for r in range(0, row):
        if img2.sum(axis=1)[r] < 700 * col:
            row_top = r
            break

    for r in range(row - 1, 0, -1):
        if img2.sum(axis=1)[r] < 700 * col:
            raw_down = r
            break

    for c in range(0, col):
        if img2.sum(axis=0)[c] < 700 * row:
            col_top = c
            break

    for c in range(col - 1, 0, -1):
        if img2.sum(axis=0)[c] < 700 * row:
            col_down = c
            break

    new_img = img[row_top:raw_down + 1, col_top:col_down + 1, 0:3]
    return new_img


def get_jet():
    colormap_int = np.zeros((256, 3), np.uint8)
    colormap_float = np.zeros((256, 3), np.float)

    for i in range(0, 256, 1):
        colormap_float[i, 0] = cm.jet(i)[0]
        colormap_float[i, 1] = cm.jet(i)[1]
        colormap_float[i, 2] = cm.jet(i)[2]

        colormap_int[i, 0] = np.int_(np.round(cm.jet(i)[0] * 255.0))
        colormap_int[i, 1] = np.int_(np.round(cm.jet(i)[1] * 255.0))
        colormap_int[i, 2] = np.int_(np.round(cm.jet(i)[2] * 255.0))

    return colormap_int

def plot_spec(spec):
    # yrange = spec.shape[0]
    # plt.ylim(-.5, yrange - .5)
    # plt.imshow(spec, cmap='jet', interpolation='nearest')
    # plt.axes().set_aspect('auto')
    # plt.savefig('pic/aaa.png')
    # a=Image.open('aaa.png')
    # print(a.size)
    # b=a.convert('RGB')#PIL=>4通道的png图像读入成jpg格式的3通道
    # print(b.size)# 只是长宽
    # print(np.array(b)-np.array(a))
    """
    cv2.imshow()采用BGR模式
    plt.imshow() 采用RGB模式
    img.show() 采用RGB模式
    """


    # a=Image.open('pic/aaa.png').convert('RGB')

    color_map=get_jet()
    # print(color_map)
    rows, cols = spec.shape
    color_array = np.zeros((rows, cols, 3), np.uint8)
    # print(color_array)
    # print(rows)
    # print(cols)
    for i in range(rows):
        for j in range(cols):
            color_array[i, j] = color_map[int(spec[i, j])]
            # print(color_array)

    # print('aa')
    # return Image.fromarray(corp_margin(np.array(a)))
    # print(color_array)
    return Image.fromarray(color_array)

def makeT(cp):
    # cp: [K x 2] control points
    # T: [(K+3) x (K+3)]
    K = cp.shape[0]
    T = np.zeros((K + 3, K + 3))
    T[:K, 0] = 1
    T[:K, 1:3] = cp
    T[K, 3:] = 1
    T[K + 1:, 3:] = cp.T
    R = squareform(pdist(cp, metric='euclidean'))
    R = R * R
    R[R == 0] = 1  # a trick to make R ln(R) 0
    R = R * np.log(R)
    np.fill_diagonal(R, 0)
    T[:K, 3:] = R
    return T


def liftPts(p, cp):
    # p: [N x 2], input points
    # cp: [K x 2], control points
    # pLift: [N x (3+K)], lifted input points
    N, K = p.shape[0], cp.shape[0]
    pLift = np.zeros((N, K + 3))
    pLift[:, 0] = 1
    pLift[:, 1:3] = p
    R = cdist(p, cp, 'euclidean')
    R = R * R
    R[R == 0] = 1
    R = R * np.log(R)
    pLift[:, 3:] = R
    return pLift

def lb_trans(input):
    audio, sampling_rate = librosa.load(input)
    spec = librosa.feature.melspectrogram(y=audio, sr=sampling_rate, n_mels=80, fmax=8000)
    spec = librosa.power_to_db(spec, ref=np.max)
    return spec


def SAWithoutAug(input):
    audio, sampling_rate = librosa.load(input,)
    spec = librosa.feature.melspectrogram(y=audio, sr=sampling_rate, n_mels=80, fmax=8000)
    spec = librosa.power_to_db(spec, ref=np.max)
    return plot_spec(spec)
# 音频转图像
# warped + masked spectrum
def SA(input,num,flag=False):

    audio, sampling_rate = librosa.load(input,)
    spec = librosa.feature.melspectrogram(y=audio, sr=sampling_rate, n_mels=80, fmax=8000)
    spec = librosa.power_to_db(spec, ref=np.max)

    if flag==True:
        return [plot_spec(spec)]    
        
    ans=[plot_spec(spec)]
    for n in range(num-1):
        start = time.time()
        W = 40
        T = 30
        F = 13
        mt = 2
        mf = 2

        # Nframe : number of spectrum frame
        Nframe = spec.shape[1]
        # Nbin : number of spectrum freq bin
        Nbin = spec.shape[0]
        # check input length
        if Nframe < W * 2 + 1:
            W = int(Nframe / 4)
        if Nframe < T * 2 + 1:
            T = int(Nframe / mt)
        if Nbin < F * 2 + 1:
            F = int(Nbin / mf)

        # warping parameter initialize
        w = random.randint(-W, W)
        center = random.randint(W, Nframe - W)

        src = np.asarray(
            [[float(center), 1], [float(center), 0], [float(center), 2], [0, 0], [0, 1], [0, 2], [Nframe - 1, 0],
             [Nframe - 1, 1], [Nframe - 1, 2]])
        dst = np.asarray(
            [[float(center + w), 1], [float(center + w), 0], [float(center + w), 2], [0, 0], [0, 1], [0, 2],
             [Nframe - 1, 0], [Nframe - 1, 1], [Nframe - 1, 2]])

        # source control points
        xs, ys = src[:, 0], src[:, 1]
        cps = np.vstack([xs, ys]).T
        # target control points
        xt, yt = dst[:, 0], dst[:, 1]
        # construct TT
        TT = makeT(cps)

        # solve cx, cy (coefficients for x and y)
        xtAug = np.concatenate([xt, np.zeros(3)])
        ytAug = np.concatenate([yt, np.zeros(3)])
        cx = nl.solve(TT, xtAug)  # [K+3]
        cy = nl.solve(TT, ytAug)

        # dense grid
        x = np.linspace(0, Nframe - 1, Nframe)
        y = np.linspace(1, 1, 1)
        x, y = np.meshgrid(x, y)

        xgs, ygs = x.flatten(), y.flatten()

        gps = np.vstack([xgs, ygs]).T

        # transform
        pgLift = liftPts(gps, cps)  # [N x (K+3)]
        xgt = np.dot(pgLift, cx.T)
        spec_warped = np.zeros_like(spec)
        for f_ind in range(Nbin):
            spec_tmp = spec[f_ind, :]
            func = interpolate.interp1d(xgt, spec_tmp, fill_value="extrapolate")
            xnew = np.linspace(0, Nframe - 1, Nframe)
            spec_warped[f_ind, :] = func(xnew)

        # sample mt of time mask ranges
        t = np.random.randint(T - 1, size=mt) + 1
        # sample mf of freq mask ranges
        f = np.random.randint(F - 1, size=mf) + 1
        # mask_t : time mask vector
        mask_t = np.ones((Nframe, 1))
        ind = 0
        t_tmp = t.sum() + mt
        for _t in t:
            k = random.randint(ind, Nframe - t_tmp)
            mask_t[k:k + _t] = 0
            ind = k + _t + 1
            t_tmp = t_tmp - (_t + 1)
        mask_t[ind:] = 1

        # mask_f : freq mask vector
        mask_f = np.ones((Nbin, 1))
        ind = 0
        f_tmp = f.sum() + mf
        for _f in f:
            k = random.randint(ind, Nbin - f_tmp)
            mask_f[k:k + _f] = 0
            ind = k + _f + 1
            f_tmp = f_tmp - (_f + 1)
        mask_f[ind:] = 1

        # calculate mean
        mean = np.mean(spec_warped)


        spec_zero = spec_warped - mean

        spec_masked = ((spec_zero * mask_t.T) * mask_f) + mean

        ans.append(plot_spec(spec_masked))

    return ans



