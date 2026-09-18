from braindecode.datasets import SleepPhysionet
from braindecode.preprocessing import preprocess, Preprocessor
from braindecode.preprocessing import create_windows_from_events
import numpy as np
from braindecode.preprocessing.preprocess import preprocess, Preprocessor


from numpy import multiply
from braindecode.preprocessing.windowers import create_windows_from_events
from sklearn.preprocessing import scale as standard_scale

import numpy as np
import torch
import os

n_jobs = 4


high_cut_hz = 30

factor = 1e6


preprocessors = [
    Preprocessor(lambda data: multiply(data, factor)),  
    Preprocessor('filter', l_freq=None, h_freq=high_cut_hz, n_jobs=n_jobs)
]






window_size_s = 30
sfreq = 100
window_size_samples = window_size_s * sfreq

mapping = {  
    'Sleep stage W': 0,
    'Sleep stage 1': 1,
    'Sleep stage 2': 2,
    'Sleep stage 3': 3,
    'Sleep stage 4': 3,
    'Sleep stage R': 4
}



dataset_fold = "./sleep_edf/"

train_dataset_fold = dataset_fold + "TrainFold/"
valid_dataset_fold = dataset_fold + "ValidFold/"
test_dataset_fold = dataset_fold + "TestFold/"



np.random.seed(7)
for sub in range(39,83): 
    if sub in [39, 68, 69, 78, 79]: continue
    r = np.random.rand()
    if r<0.6: 
        save_path = train_dataset_fold
    elif r<0.8:
        save_path = valid_dataset_fold
    else:
        save_path = test_dataset_fold
        
    dataset = SleepPhysionet(subject_ids=[sub], 
        crop_wake_mins=30)

    preprocess(dataset, preprocessors)
    windows_dataset = create_windows_from_events(
        dataset, trial_start_offset_samples=0, trial_stop_offset_samples=0,
        window_size_samples=window_size_samples,
        window_stride_samples=window_size_samples, preload=True, mapping=mapping)

    preprocess(windows_dataset, [Preprocessor(standard_scale, channel_wise=True)])
    for i, x in enumerate(windows_dataset):
        path = save_path+str(x[1])+'/'
        os.makedirs(path, exist_ok=True)
        path+= f"s{sub}_{x[2][1]}_{x[2][2]}.pt"
        torch.save(torch.tensor(x[0]), path)
        
