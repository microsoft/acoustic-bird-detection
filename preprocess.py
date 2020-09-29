import glob

import numpy as np
import pandas as pd

import soundfile

from python_speech_features import fbank
from scipy.signal import spectrogram
from os import path

def rescale_for_model(fbanks):
    pass


def make_fbank(wav, fs=22050):
    winlen = 1. / 43.0664 # specRes_Hz from model 
    winstep = 2.9 / 1000. # tempRes_ms from model
    nfft = 1024
    preemph = 0.5
    M, _ = fbank(wav, samplerate=fs,
                 nfilt=41, nfft=nfft,
                 lowfreq=0, highfreq=11025,
                 preemph=0.5,
                 winlen=winlen, winstep=winstep,
                 winfunc=lambda x: np.hanning(x))

    logM = np.log(M)
    logM = np.swapaxes(logM, 0, 1)

    targetSize = 682
    cut = np.minimum(logM.shape[1], targetSize)
    background = np.float64(logM[:,:cut]).mean(axis=1)

    features = np.float32(np.float64(logM) - background[:, np.newaxis])

    if features.shape[1] < targetSize:
        features = np.concatenate((features,
                                   np.zeros((features.shape[0],
                                             targetSize-features.shape[1]),
                                            dtype='float32')), axis=1)
    elif features.shape[1] > targetSize:
        features = features[:,:(targetSize-features.shape[1])]

    return features

def preprocess_data():
    window_len = 2 # seconds

    # Remove any positive label from a window where the call is shorter than this.
    minimum_call_seconds = 0.2 # seconds

    target = np.array([], dtype=np.int8)
    data = None

    df = pd.read_csv("resources/ANTBOK_training_labels.csv")
    df["duration"] = df["end.time"] - df["start.time"]
    csv = df[df["duration"] > minimum_call_seconds]

    print("processing ...")
    for f in glob.glob("resources/training/clips/*.flac"):
        s = path.splitext(path.basename(f))[0].replace("_", "-") + '.mp3'
        boxes = csv[csv['filename'] == s]
    
        # read the audio file. if this is a stereo file, 
        # we take only the first channel.
        wav, fs = soundfile.read(f)
        if np.ndim(wav) > 1:
            wav = wav[:,0]
        samples_per_window = fs * window_len
    
        # generate all 2 seconds windows as a list, then round down
        # the label start time to the nearest 2 second window.
        all_windows = np.arange(
            0, np.ceil(len(wav) / samples_per_window).astype(np.int))

        positive_windows = ((np.floor(boxes['start.time']).astype(
            np.int) * fs) // samples_per_window).to_numpy()
    
        for w in all_windows:
            if w in positive_windows:
                target = np.append(target, 1)
            else:
                target = np.append(target, 0)
        
            start = w * samples_per_window
            end = start + samples_per_window
                     
            window = wav[start:end]
            
            if data is None:
                data = np.array([[window, f, w]])
            else:
                data = np.append(data, [[window, f, w]], axis=0)
    return data, target
