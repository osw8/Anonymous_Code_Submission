import mne
import numpy as np
import os
import scipy.io as sio


channel_names = ['FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1', 'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1', 'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2', 'FP2-F8', 'F8-T8', 'T8-P8-0', 'P8-O2', 'FZ-CZ', 'CZ-PZ', 'T8-P8-1', 'FC1-Ref', 'FC2-Ref', 'FC5-Ref', 'FC6-Ref']


data_dir = 'F:/chb-mit-scalp-eeg-database-1.0.0/chb06/'


segments = {
    'chb06_01.edf': (1724, 1738),
    'chb06_01.edf': (7461, 7476),
    'chb06_01.edf': (13525, 13540),
    'chb06_04.edf': (327, 347),
    'chb06_04.edf': (6211, 6231),
    'chb06_09.edf': (12500, 12516),
    'chb06_10.edf': (10833, 10845),
    'chb06_13.edf': (506, 519),
    'chb06_18.edf': (7799, 7811),
    'chb06_24.edf': (9387, 9403),

}



nonseizure_segment = (0, 3600)


seizure_data = []
nonseizure_data = []


for file, segment in segments.items():
    file_path = os.path.join(data_dir, file)
    raw = mne.io.read_raw_edf(file_path, preload=True, verbose='error')
    raw.pick_channels(channel_names)

    
    start, stop = segment
    start = int(start * raw.info['sfreq'])
    stop = int(stop * raw.info['sfreq'])

    
    seizure_data.append(raw._data[:, start:stop] * 1e6)  


seizure_data = np.concatenate(seizure_data, axis=1)
seizure_data = seizure_data.T


file_path = os.path.join(data_dir, 'chb06_02.edf')
raw = mne.io.read_raw_edf(file_path, preload=True, verbose='error')
raw.pick_channels(channel_names)
start, stop = nonseizure_segment
start = int(start * raw.info['sfreq'])
stop = int(stop * raw.info['sfreq'])
nonseizure_data = raw._data[:, start:stop] * 1e6  


nonseizure_data = np.array(nonseizure_data)
nonseizure_data = nonseizure_data.T

sio.savemat('F:/solver ep/data6.mat', {'seizure_data': seizure_data, 'nonseizure_data': nonseizure_data})
