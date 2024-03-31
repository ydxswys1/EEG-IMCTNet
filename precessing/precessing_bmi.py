import scipy.io
import numpy as np
import matplotlib
import mne
import os, sys
from scipy import signal
import warnings
from mne.preprocessing import ICA,  create_eog_epochs
warnings.filterwarnings("ignore")
matplotlib.use('Qt5Agg')
def filterd(root,name_train,savapath):
    filename_train=os.path.join(root,name_train)
    raw_eeg_subject_m = scipy.io.loadmat(filename_train)
    data=raw_eeg_subject_m['data']
    label=raw_eeg_subject_m['label']
    fs = 250  # 采样率，单位为Hz
    lowcut = 4  # 低频截止频率，单位为Hz
    highcut = 40.0  # 高频截止频率，单位为Hz
    nyquist = 0.5 * fs

    # 计算滤波器的截止频率在Nyquist频率中的比例
    low = lowcut / nyquist
    high = highcut / nyquist

    # 创建一个切比雪夫带通滤波器
    N = 6  # 滤波器的阶数
    rp = 0.5  # 通带纹波
    rs = 30  # 阻带衰减
    b, a = signal.cheby2(N, rs, [low, high], btype='bandpass')

    filtered_eeg_data = np.zeros_like(data)

    # 对每个批次中的每个通道的数据应用滤波器
    for batch_idx in range(data.shape[0]):
        for channel_idx in range(data.shape[1]):
            filtered_eeg_data[batch_idx, channel_idx, :] = signal.lfilter(b, a, data[batch_idx, channel_idx, :])

    # filtered_eeg_data 中包含了滤波后的EEG数据
    data = {'data': filtered_eeg_data[:, :, :1000], 'label': label.reshape(-1, 1)}
    scipy.io.savemat(os.path.join(savapath, name_train), data)

root=r'D:\download\main\dataset\openBMI'
savapath = r'D:\download\main\dataset\filterd_bmi'
# for i in range(27,55):
for i in [25]:
    name=f'A{i}'
    print(name)
    name_train = name + str("T.mat")
    name_test = name + str("E.mat")
    filterd(root,name_train,savapath)
    filterd(root,name_test,savapath)
