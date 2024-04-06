import mne
import numpy as np
import torch
import torch.nn as nn
import os
import scipy.io as sio
import matplotlib

matplotlib.use('Qt5Agg')


def pre(root_x, root_y, name_x, name_y, exist_question=False, test=False):
    filename_x = os.path.join(root_x, name_x)
    filename_y = os.path.join(root_y, name_y)

    raw = mne.io.read_raw_gdf(filename_x)

    events, events_dict = mne.events_from_annotations(raw)
    print(events)
    print(events_dict)
    raw.load_data()

    # raw.filter(4., 40., fir_design='firwin')

    raw.info['bads'] += ['EOG:ch01', 'EOG:ch02', 'EOG:ch03']
    picks = mne.pick_types(raw.info, meg=False, eeg=True, eog=False, stim=False,
                           exclude='bads')

    tmin, tmax = 0., 4.
    event_id = dict({'769': 10, '770': 11})
    if exist_question:
        event_id = dict({'769': 4, '770': 5})
    if test:
        event_id = dict({'783': 11})
    if exist_question==True and test==True:
        event_id = dict({'783': 5})
    epochs = mne.Epochs(raw, events, event_id, tmin, tmax, proj=True, picks=picks,
                        baseline=None, preload=True)
    x_data = epochs.get_data() * 1e6
    x_data = x_data[:, :, :1000]

    # y_label=epochs.events[:, -1]- min(epochs.events[:, -1])
    y_label = sio.loadmat(filename_y)["classlabel"]
    # epochs.plot()
    return x_data, y_label



# Note:The B0102T.gdf of the dataset subject2 and B504E.gdf of the dataset subject5 save the data in a different format than the others
#
# catalogue:
#     \BCICIV_2b_gdf
#         \B010T.gdf...
#     \true_labels
#         \B010T.mat...


root_x = r"BCI_2B\BCICIV_2b_gdf"
root_y = r"BCI_2B\true_labels"
for w in range(1,10):
    for i in range(1, 6):

        name = f"B0{w}0{i}"
        if i < 4:
            name_x = name + str("T.gdf")
            name_y = name + str("T.mat")
            test = False
        else:
            name_x = name + str("E.gdf")
            name_y = name + str("E.mat")
            test = True
        exist_question=False

        if w==1:
            if i == 2:
                exist_question = True
            else:
                exist_question = False
        if w==5:
            if i == 4:
                exist_question = True
            else:
                exist_question = False
        # print(name)
        print("11111111111111111111111111")
        print(test)
        print(exist_question)
        x, y = pre(root_x, root_y, name_x, name_y, test=test, exist_question=exist_question)

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

    print(x_train.shape)
    print(y_train.shape)
    print(x_test.shape)
    print(y_test.shape)

    data = {'data': x_train[:, :, :1000], 'label': y_train.reshape(-1, 1)}
    import scipy.io

    print(data['data'].shape)

    scipy.io.savemat(rf'//////Data\A0{w}T.mat', data)#Save the path of the preprocessed data

    data1 = {'data': x_test[:, :, :1000], 'label': y_test.reshape(-1, 1)}
    import scipy.io

    print(data1['data'].shape)

    scipy.io.savemat(rf'///////\Data\A0{w}E.mat', data1)#Save the path of the preprocessed data