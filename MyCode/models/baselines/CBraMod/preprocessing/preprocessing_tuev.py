import mne
import numpy as np
import os
import pickle
from tqdm import tqdm



def BuildEvents(signals, times, EventData):
    [numEvents, z] = EventData.shape  
    fs = 200.0
    [numChan, numPoints] = signals.shape
    
    
    
    features = np.zeros([numEvents, numChan, int(fs) * 5])
    offending_channel = np.zeros([numEvents, 1])  
    labels = np.zeros([numEvents, 1])
    offset = signals.shape[1]
    signals = np.concatenate([signals, signals, signals], axis=1)
    for i in range(numEvents):  
        chan = int(EventData[i, 0])  
        start = np.where((times) >= EventData[i, 1])[0][0]
        end = np.where((times) >= EventData[i, 2])[0][0]
        
        features[i, :] = signals[
            :, offset + start - 2 * int(fs) : offset + end + 2 * int(fs)
        ]
        offending_channel[i, :] = int(chan)
        labels[i, :] = int(EventData[i, 3])
    return [features, offending_channel, labels]


def convert_signals(signals, Rawdata):
    signal_names = {
        k: v
        for (k, v) in zip(
            Rawdata.info["ch_names"], list(range(len(Rawdata.info["ch_names"])))
        )
    }
    new_signals = np.vstack(
        (
            signals[signal_names["EEG FP1-REF"]]
            - signals[signal_names["EEG F7-REF"]],  
            (
                signals[signal_names["EEG F7-REF"]]
                - signals[signal_names["EEG T3-REF"]]
            ),  
            (
                signals[signal_names["EEG T3-REF"]]
                - signals[signal_names["EEG T5-REF"]]
            ),  
            (
                signals[signal_names["EEG T5-REF"]]
                - signals[signal_names["EEG O1-REF"]]
            ),  
            (
                signals[signal_names["EEG FP2-REF"]]
                - signals[signal_names["EEG F8-REF"]]
            ),  
            (
                signals[signal_names["EEG F8-REF"]]
                - signals[signal_names["EEG T4-REF"]]
            ),  
            (
                signals[signal_names["EEG T4-REF"]]
                - signals[signal_names["EEG T6-REF"]]
            ),  
            (
                signals[signal_names["EEG T6-REF"]]
                - signals[signal_names["EEG O2-REF"]]
            ),  
            (
                signals[signal_names["EEG FP1-REF"]]
                - signals[signal_names["EEG F3-REF"]]
            ),  
            (
                signals[signal_names["EEG F3-REF"]]
                - signals[signal_names["EEG C3-REF"]]
            ),  
            (
                signals[signal_names["EEG C3-REF"]]
                - signals[signal_names["EEG P3-REF"]]
            ),  
            (
                signals[signal_names["EEG P3-REF"]]
                - signals[signal_names["EEG O1-REF"]]
            ),  
            (
                signals[signal_names["EEG FP2-REF"]]
                - signals[signal_names["EEG F4-REF"]]
            ),  
            (
                signals[signal_names["EEG F4-REF"]]
                - signals[signal_names["EEG C4-REF"]]
            ),  
            (
                signals[signal_names["EEG C4-REF"]]
                - signals[signal_names["EEG P4-REF"]]
            ),  
            (signals[signal_names["EEG P4-REF"]] - signals[signal_names["EEG O2-REF"]]),
        )
    )  
    return new_signals


def readEDF(fileName):
    Rawdata = mne.io.read_raw_edf(fileName, preload=True)
    Rawdata.resample(200)
    Rawdata.filter(l_freq=0.3, h_freq=75)
    Rawdata.notch_filter((60))

    _, times = Rawdata[:]
    signals = Rawdata.get_data(units='uV')
    RecFile = fileName[0:-3] + "rec"
    eventData = np.genfromtxt(RecFile, delimiter=",")
    Rawdata.close()
    return [signals, times, eventData, Rawdata]


def load_up_objects(BaseDir, Features, OffendingChannels, Labels, OutDir):
    for dirName, subdirList, fileList in tqdm(os.walk(BaseDir)):
        print("Found directory: %s" % dirName)
        for fname in fileList:
            if fname[-4:] == ".edf":
                print("\t%s" % fname)
                try:
                    [signals, times, event, Rawdata] = readEDF(
                        dirName + "/" + fname
                    )  
                    signals = convert_signals(signals, Rawdata)
                except (ValueError, KeyError):
                    print("something funky happened in " + dirName + "/" + fname)
                    continue
                signals, offending_channels, labels = BuildEvents(signals, times, event)

                for idx, (signal, offending_channel, label) in enumerate(
                    zip(signals, offending_channels, labels)
                ):
                    sample = {
                        "signal": signal,
                        "offending_channel": offending_channel,
                        "label": label,
                    }
                    save_pickle(
                        sample,
                        os.path.join(
                            OutDir, fname.split(".")[0] + "-" + str(idx) + ".pkl"
                        ),
                    )

    return Features, Labels, OffendingChannels


def save_pickle(object, filename):
    with open(filename, "wb") as f:
        pickle.dump(object, f)



root = "data/TUEV/edf"
target = "data/datasets/BigDownstream/TUEV_refine"

train_out_dir = os.path.join(target, "processed_train")
eval_out_dir = os.path.join(target, "processed_eval")

if not os.path.exists(train_out_dir):
    os.makedirs(train_out_dir)
if not os.path.exists(eval_out_dir):
    os.makedirs(eval_out_dir)

BaseDirTrain = os.path.join(root, "train")
fs = 200
TrainFeatures = np.empty(
    (0, 16, fs)
)  
TrainLabels = np.empty([0, 1])
TrainOffendingChannel = np.empty([0, 1])
load_up_objects(
    BaseDirTrain, TrainFeatures, TrainLabels, TrainOffendingChannel, train_out_dir
)

BaseDirEval = os.path.join(root, "eval")
fs = 200
EvalFeatures = np.empty(
    (0, 16, fs)
)  
EvalLabels = np.empty([0, 1])
EvalOffendingChannel = np.empty([0, 1])
load_up_objects(
    BaseDirEval, EvalFeatures, EvalLabels, EvalOffendingChannel, eval_out_dir
)



root = "data/datasets/BigDownstream/TUEV_refine"



train_files = os.listdir(os.path.join(root, "processed_train"))
train_val_sub = list(set([f.split("_")[0] for f in train_files]))
print("train val sub:", train_val_sub)
test_files = os.listdir(os.path.join(root, "processed_eval"))

train_val_sub.sort(key=lambda x: x)

train_sub = train_val_sub[: int(len(train_val_sub) * 0.8)]
val_sub = train_val_sub[int(len(train_val_sub) * 0.8) :]
print("train sub:", train_sub)
print("val sub:", val_sub)

val_files = [f for f in train_files if f.split("_")[0] in val_sub]
train_files = [f for f in train_files if f.split("_")[0] in train_sub]


if not os.path.exists(os.path.join(root, 'processed', 'processed_train')):
    os.makedirs(os.path.join(root, 'processed', 'processed_train'))
if not os.path.exists(os.path.join(root, 'processed', 'processed_eval')):
    os.makedirs(os.path.join(root, 'processed', 'processed_eval'))
if not os.path.exists(os.path.join(root, 'processed', 'processed_test')):
    os.makedirs(os.path.join(root, 'processed', 'processed_test'))

for file in tqdm(train_files):
    os.system(f"cp {os.path.join(root, 'processed_train', file)} {os.path.join(root, 'processed', 'processed_train')}")
for file in tqdm(val_files):
    os.system(f"cp {os.path.join(root, 'processed_train', file)} {os.path.join(root, 'processed', 'processed_eval')}")
for file in tqdm(test_files):
    os.system(f"cp {os.path.join(root, 'processed_eval', file)} {os.path.join(root, 'processed', 'processed_test')}")

print('Done!')
