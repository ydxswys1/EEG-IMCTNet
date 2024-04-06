import scipy.io as io
import resampy
import scipy.io
import numpy as np
import matplotlib
import os, sys
from scipy import signal
import warnings
from mne.preprocessing import ICA,  create_eog_epochs
warnings.filterwarnings("ignore")
matplotlib.use('Qt5Agg')

# from data_utils import *

def fetchKoreaDataFile(dataPath, epochWindow=[0, 4], chans=None, downsampleFactor=None):
    alldata = io.loadmat(dataPath)

    data = np.concatenate((alldata['EEG_MI_train'][0, 0]['smt'], alldata['EEG_MI_test'][0, 0]['smt']), axis=1)
    labels = np.concatenate(
        (alldata['EEG_MI_train'][0, 0]['y_dec'].squeeze(), alldata['EEG_MI_test'][0, 0]['y_dec'].squeeze())).astype(
        int) - 1

    allchans = np.array([m.item() for m in alldata['EEG_MI_train'][0, 0]['chan'].squeeze()])
    fs = alldata['EEG_MI_train'][0, 0]['fs'].squeeze().item()

    del alldata

    if chans is not None:
        data = data[:, :, chans]
        allchans = allchans[np.array(chans)]

    if downsampleFactor is not None:
        # dataNew = np.zeros((int(data.shape[0]/downsampleFactor), *data.shape[1:3]), np.float)
        dataNew = np.zeros((int(data.shape[0] / downsampleFactor), *data.shape[1:3]), float)

        for i in range(data.shape[2]):
            dataNew[:, :, i] = resampy.resample(data[:, :, i], fs, fs // downsampleFactor, axis=0)
        data = dataNew
        fs = fs // downsampleFactor

    if epochWindow != [0, 4]:
        start = epochWindow[0] * fs
        end = epochWindow[1] * fs
        data = data[start:end, :, :]

    # change the data dimension: trials x channels x time
    data = np.transpose(data, axes=(1, 2, 0))

    return {'data': data, 'label': labels.reshape(-1, 1) + 1, 'chans': allchans, 'fs': fs}


def preprocessKoreaDataset(datasetPath, savePath, epochWindow=[0, 4],
                           chans=[7, 32, 8, 9, 33, 10, 34, 12, 35, 13, 36, 14, 37, 17, 38, 18, 39, 19, 40, 20],
                           downsampleFactor=4):
    subjects = list(range(54))
    if not os.path.exists(savePath):
        os.makedirs(savePath)
    print('Processed raw.gdf data will be saved in folder', savePath)

    for sub in subjects:
        print(f'Processing subject No. {sub + 1}'.format(sub))
        trainFile = os.path.join(datasetPath, 'Sess01', 'sess01_subj' + str(sub + 1).zfill(2) + '_EEG_MI.mat')
        testFile = os.path.join(datasetPath, 'Sess02', 'sess02_subj' + str(sub + 1).zfill(2) + '_EEG_MI.mat')
        print(trainFile)
        assert (os.path.exists(trainFile) and os.path.exists(testFile)), 'Do not find data, check the data path...'

        trainData = fetchKoreaDataFile(trainFile, epochWindow=epochWindow, chans=chans,
                                       downsampleFactor=downsampleFactor)
        testData = fetchKoreaDataFile(testFile, epochWindow=epochWindow, chans=chans, downsampleFactor=downsampleFactor)

        io.savemat(os.path.join(savePath, 'A' + str(sub + 1).zfill(2) + 'T.mat'), trainData)
        io.savemat(os.path.join(savePath, 'A' + str(sub + 1).zfill(2) + 'E.mat'), testData)

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
    print(label.shape)
    data = {'data': filtered_eeg_data[:, :, :1000], 'label': label.reshape(-1, 1)}
    scipy.io.savemat(os.path.join(savapath, name_train), data)


# catalogue:
#     \datasetPath:
#         \Sess01:
#             sess01_subj01_EEG_MI.mat...
#         \Sess02:
#             sess02_subj01_EEG_MI.mat...
# semi_finished_savePath is a path that holds pre-processed semi-finished products
# savapath is the path to save the preprocessed data

if __name__ == '__main__':
    datasetPath = 'D:\download\LightConvNet-main\dataset\OpenBMI_MAT'
    semi_finished_savePath = r'E:\project\pythonProject\semi_finished_savePath'
    preprocessKoreaDataset(datasetPath, semi_finished_savePath)
    savapath = r'E:\project\pythonProject\BMI'
    for i in range(1,55):

        name='A'+str(i).zfill(2)
        print(name)
        name_train = name + str("T.mat")
        name_test = name + str("E.mat")
        filterd(semi_finished_savePath,name_train,savapath)
        filterd(semi_finished_savePath,name_test,savapath)
