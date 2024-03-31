import mne
import numpy as np
import torch
import torch.nn as nn
import os
import scipy.io as sio
import matplotlib

matplotlib.use('Qt5Agg')

filename = r"E:\notebook\pyproject\EEG-Motor-Imagery-Classification---ANN-master\BAC_2B\BCICIV_2b_gdf\B0101T.gdf"


def pre(root_x, root_y, name_x, name_y, exist_question=False, test=False):
    filename_x = os.path.join(root_x, name_x)
    filename_y = os.path.join(root_y, name_y)

    raw = mne.io.read_raw_gdf(filename_x)

    events, _ = mne.events_from_annotations(raw)

    raw.load_data()

    raw.filter(4., 40., fir_design='firwin')

    raw.info['bads'] += ['EOG:ch01', 'EOG:ch02', 'EOG:ch03']
    picks = mne.pick_types(raw.info, meg=False, eeg=True, eog=False, stim=False,
                           exclude='bads')

    tmin, tmax = 0., 4.

    event_id = dict({'769': 10, '770': 11})
    if exist_question:
        event_id = dict({'769': 4, '770': 5})
    if test:
        event_id = dict({'783': 11})
    epochs = mne.Epochs(raw, events, event_id, tmin, tmax, proj=True, picks=picks,
                        baseline=None, preload=True)
    x_data = epochs.get_data() * 1e6
    x_data = x_data[:, :, :1000]

    # y_label=epochs.events[:, -1]- min(epochs.events[:, -1])
    y_label = sio.loadmat(filename_y)["classlabel"]
    # epochs.plot()
    return x_data, y_label


root_x = r"E:\notebook\pyproject\EEG-Motor-Imagery-Classification---ANN-master\BAC_2B\BCICIV_2b_gdf"
root_y = r"E:\notebook\pyproject\EEG-Motor-Imagery-Classification---ANN-master\BAC_2B\true_labels"
# x,y=pre(root_x,root_y,"B0205E.gdf","B0205E.mat",test=True)
# print(y.shape)
for i in range(1, 6):

    name = f"B010{i}"
    if i < 4:
        name_x = name + str("T.gdf")
        name_y = name + str("T.mat")
        test = False
    else:
        name_x = name + str("E.gdf")
        name_y = name + str("E.mat")
        test = True
    if i == 2:
        exist_question = True
    else:
        exist_question = False
    # print(name)
    x, y = pre(root_x, root_y, name_x, name_y, test=False, exist_question=exist_question)

    if i == 1:
        x_train = x
        y_train = y
    if (1 < i) and (i < 4):
        x_train = np.concatenate((x_train, x), axis=0)
        y_train = np.concatenate((y_train, y), axis=0)
    if i == 4:
        x_test = x
        y_test = y
    if (4 < i) and (i < 6):
        x_test = np.concatenate((x_test, x), axis=0)
        y_test = np.concatenate((y_test, y), axis=0)

data = {'data': x_train[:, :, :1000], 'label': y_train.reshape(-1, 1) }
import scipy.io

# 使用savemat函数保存数据
scipy.io.savemat('A04T.mat', data)

data = {'data': x_test[:, :, :1000], 'label': y_test.reshape(-1, 1)}
import scipy.io

# 使用savemat函数保存数据
scipy.io.savemat('A04E.mat', data)