import pickle
import os
import numpy as np
from tqdm import tqdm
import multiprocessing as mp

root = "data/physionet.org/files/chbmit/1.0.0/clean_signals"
out = "data/physionet.org/files/chbmit/1.0.0/clean_segments"




if not os.path.exists(out):
    os.makedirs(out)


test_pats = ["chb23", "chb24"]
val_pats = ["chb21", "chb22"]
train_pats = [
    "chb01",
    "chb02",
    "chb03",
    "chb04",
    "chb05",
    "chb06",
    "chb07",
    "chb08",
    "chb09",
    "chb10",
    "chb11",
    "chb12",
    "chb13",
    "chb14",
    "chb15",
    "chb16",
    "chb17",
    "chb18",
    "chb19",
    "chb20",
]
channels = [
    "FP1-F7",
    "F7-T7",
    "T7-P7",
    "P7-O1",
    "FP2-F8",
    "F8-T8",
    "T8-P8",
    "P8-O2",
    "FP1-F3",
    "F3-C3",
    "C3-P3",
    "P3-O1",
    "FP2-F4",
    "F4-C4",
    "C4-P4",
    "P4-O2",
]
SAMPLING_RATE = 256


def sub_to_segments(folder, out_folder):
    print(f"Processing {folder}...")
    
    for f in tqdm(os.listdir(os.path.join(root, folder))):
        print(f"Processing {folder}/{f}...")
        record = pickle.load(open(os.path.join(root, folder, f), "rb"))
        signal = []
        for channel in channels:
            if channel in record:
                signal.append(record[channel])
            else:
                raise ValueError(f"Channel {channel} not found in record {record}")
        signal = np.array(signal)

        if "times" in record["metadata"]:
            seizure_times = record["metadata"]["times"]
        else:
            seizure_times = []

        
        for i in range(0, signal.shape[1], SAMPLING_RATE * 10):
            segment = signal[:, i : i + 10 * SAMPLING_RATE]
            if segment.shape[1] == 10 * SAMPLING_RATE:
                
                label = 0

                for seizure_time in seizure_times:
                    if (
                        i < seizure_time[0] < i + 10 * SAMPLING_RATE
                        or i < seizure_time[1] < i + 10 * SAMPLING_RATE
                    ):
                        label = 1
                        break

                
                pickle.dump(
                    {"X": segment, "y": label},
                    open(
                        os.path.join(out_folder, f"{f.split('.')[0]}-{i}.pkl"),
                        "wb",
                    ),
                )

        for idx, seizure_time in enumerate(seizure_times):
            for i in range(
                max(0, seizure_time[0] - SAMPLING_RATE),
                min(seizure_time[1] + SAMPLING_RATE, signal.shape[1]),
                5 * SAMPLING_RATE,
            ):
                segment = signal[:, i : i + 10 * SAMPLING_RATE]
                label = 1
                
                pickle.dump(
                    {"X": segment, "y": label},
                    open(
                        os.path.join(
                            out_folder, f"{f.split('.')[0]}-s-{idx}-add-{i}.pkl"
                        ),
                        "wb",
                    ),
                )



folders = os.listdir(root)
out_folders = []
for folder in folders:
    if folder in test_pats:
        out_folder = os.path.join(out, "test")
    elif folder in val_pats:
        out_folder = os.path.join(out, "val")
    else:
        out_folder = os.path.join(out, "train")

    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    out_folders.append(out_folder)


with mp.Pool(mp.cpu_count()) as pool:
    res = pool.starmap(sub_to_segments, zip(folders, out_folders))
