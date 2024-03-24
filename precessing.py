import mne
import numpy as np
import torch
import torch.nn as nn
import os

import matplotlib

matplotlib.use('Qt5Agg')
import glob

import numpy as np
import random
import scipy.signal as signal
import scipy.io as io
import os
import resampy
class LoadData:
    def __init__(self, eeg_file_path: str):
        self.eeg_file_path = eeg_file_path

    def load_raw_data_gdf(self, file_to_load):
        self.raw_eeg_subject = mne.io.read_raw_gdf(self.eeg_file_path + '/' + file_to_load)
        return self

    def load_raw_data_mat(self, file_to_load):
        import scipy.io as sio
        self.raw_eeg_subject_m = sio.loadmat(self.eeg_file_path + '/' + file_to_load)
        return self

    def get_all_files(self, file_path_extension: str = ''):
        if file_path_extension:
            return glob.glob(self.eeg_file_path + '/' + file_path_extension)
        return os.listdir(self.eeg_file_path)


class LoadBCIC(LoadData):
    '''Subclass of LoadData for loading BCI Competition IV Dataset 2a'''

    def __init__(self, file_to_load, *args):
        self.stimcodes = ('769', '770', '771', '772')
        # self.epoched_data={}
        self.file_to_load = file_to_load
        self.channels_to_remove = ['EOG-left', 'EOG-central', 'EOG-right']
        super(LoadBCIC, self).__init__(*args)

    def get_epochs(self, tmin=1, tmax=4, baseline=None, test=None, four=None):  # 0,1
        self.load_raw_data_gdf(self.file_to_load)
        raw_data = self.raw_eeg_subject
        # raw_data.plot()
        # raw_downsampled = raw_data.copy().resample(sfreq=128)
        raw_data.load_data()
        raw_data.filter(5., 40., fir_design='firwin')
        self.fs = raw_data.info.get('sfreq')
        events, event_ids = mne.events_from_annotations(raw_data)
        stims = [value for key, value in event_ids.items() if key in self.stimcodes]
        # stims =[7]
        # epochs = mne.Epochs(raw_data, events, event_id=stims, tmin=tmin, tmax=tmax, event_repeated='drop',
        #                     baseline=baseline, preload=True, proj=False, reject_by_annotation=False)

        tmin, tmax = 0., 4.
        # left_hand = 769,right_hand = 770,foot = 771,tongue = 772
        if test == None:
            event_id = dict({'769': 7, '770': 8, '771': 9, '772': 10})
            if four == True:
                event_id = dict({'769': 5, '770': 6, '771': 7, '772': 8})
        else:
            event_id = dict({'783': 7})
        # event_id = dict({'783': 7})
        epochs = mne.Epochs(raw_data, events, event_id, tmin, tmax, proj=True, event_repeated='drop',
                            reject_by_annotation=False,
                            baseline=baseline, preload=True)
        epochs = epochs.drop_channels(self.channels_to_remove)
        # epochs.plot()
        self.y_labels = epochs.events[:, -1] - min(epochs.events[:, -1])
        self.x_data = epochs.get_data() * 1e6
        eeg_data = {'x_data': self.x_data,
                    'y_labels': self.y_labels,
                    'fs': self.fs}
        return eeg_data

    def get_label(self):
        self.load_raw_data_mat(self.file_to_load)
        label = self.raw_eeg_subject_m
        label = label['classlabel'] - 1
        return label.reshape(-1)


data_path = r"D:\download\TSFCNet-main\data\bci42a\originalData"
for w in range(1, 10):
    if w == 4:
        four = True
    else:
        four = None
    file_to_load_train = f'A0{str(w)}T.gdf'
    file_to_load_test = f'A0{str(w)}E.gdf'
    file_to_load_label = f'A0{str(w)}E.mat'

    '''for BCIC Dataset'''
    bcic_data_train = LoadBCIC(file_to_load_train, data_path)
    bcic_data_test = LoadBCIC(file_to_load_test, data_path)
    bcic_data_label = LoadBCIC(file_to_load_label, data_path)
    # bcic_data_label = LoadBCIC(data_path, file_to_load_label)
    eeg_data_train = bcic_data_train.get_epochs(four=four)

    eeg_data_test = bcic_data_test.get_epochs(test=True)  # {'x_data':, 'y_labels':, 'fs':}
    eeg_data_label = bcic_data_label.get_label()
    eeg_data_test['y_labels'] = eeg_data_label
    # eeg_label=bcic_data_label.get_label()

    # # eeg_label=bcic_data_label.get_label()
    # print(eeg_data_train)
    # print(eeg_data_test)
    import scipy.io

    X_train = eeg_data_train.get('x_data')
    Y_train = eeg_data_train.get('y_labels')
    Y_train.shape, X_train.shape
    X_test = eeg_data_test.get('x_data')
    Y_test = eeg_data_test.get('y_labels')
    Y_test.shape, X_test.shape

    print(X_train.shape)
    print(X_test.shape)

    X_train = X_train[:, :, :1000]
    X_test = X_test[:, :, :1000]

    # final_data['data'] = multifreq_train_data
    # final_data_test['data'] = multifreq_test_data

    data = {'data': X_train, 'label': Y_train.reshape(288, 1) + 1}
    scipy.io.savemat(fr'E:\project\EEG-IMCTNet\Data\test_data\A0{str(w)}T.mat', data)
    data = {'data': X_test, 'label': Y_test.reshape(288, 1) + 1}
    scipy.io.savemat(fr'E:\project\EEG-IMCTNet\Data\test_data\A0{str(w)}E.mat', data)