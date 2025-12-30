# -*- coding: utf-8 -*-
"""
@author: P Xia
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from datasets.SequenceDatasets import dataset
from datasets.sequence_aug import *
from tqdm import tqdm
import scipy.io as io
import wave

frequency = 5120
signal_size = 1024
overlap = 0
start_time = 10
num_per_cat = 1000

Category = ['HE', 'BB', 'SW', 'MP', 'BE', 'BO', 'MA', 'UN']

Label = [i for i in range(len(Category))]

usecols_vib = [4, 5, 6] # Vibration Signal
usecols_cur = [7, 8, 9] # Current Signal
usecols = usecols_vib + usecols_cur

#working condition
WC = {0:"MS20_LS",
      1:"MS30_LS",
      2:"MS40_LS",
      3:"MS45_LS",
      4:"MS30_LV2",
      5:"MS30_LV20",
      6:"MV2.5_LS",
      7:"MV5_LS",
      8:"MV10_LS"}

#generate Training Dataset and Testing Dataset
def get_files(root, N, signal_size):
    '''
    This function is used to generate the final training set and test set.
    root:The location of the data set
    '''
    data = []
    lab = []
    for k in range(len(N)):
        state1 = WC[N[k]]  # WC[0] can be changed to different working states
        for i in tqdm(range(len(Label))):
            root1 = os.path.join(root, Category[i])
            datalist1 = os.listdir(root1)
            path1 = ''
            for j, fn in enumerate(datalist1):
                if state1 in fn:
                    path1 = os.path.join(root1, fn)
            data1, lab1 = data_load(path1,label=Label[i],usecols=usecols,signal_size=signal_size,num_per_cat=num_per_cat)
            print('{} has been loaded'.format(path1))
            data += data1
            lab  += lab1
    return [data,lab]

def data_load(filename, label, usecols, signal_size, num_per_cat):
    '''
    This function is mainly used to generate test data and training data.
    filename:Data location
    '''
    fl = np.loadtxt(filename,delimiter = ",", skiprows = 1+int(frequency*start_time),usecols = usecols,encoding='utf-8')

    data=[]
    lab=[]
    num = 0
    start = 0
    end = start + signal_size
    while end<=fl.shape[0] and num<num_per_cat:
        x = fl[start:end]
        data.append(x)
        lab.append(label)
        start += int(signal_size*(1-overlap))
        end += int(signal_size*(1-overlap))
        num += 1

    return data, lab

# Load sensory data and audio data
def get_multimodal(root, N, signal_size=signal_size):
    '''
    Read simultaneously collected multi-sensor and audio data
    '''
    data = []
    lab = []
    for k in range(len(N)):
        state1 = WC[N[k]]
        for i in tqdm(range(len(Label))):
            # 1. Read multi-sensor data
            root1 = os.path.join(root+'/Ni_data(csv)', Category[i])
            datalist1 = os.listdir(root1)
            path1 = ''
            for fn in datalist1:
                if state1 in fn:
                    path1 = os.path.join(root1, fn)
            vibration_data, labels = data_load(path1, label=Label[i], usecols=usecols_vib, signal_size=signal_size, num_per_cat=num_per_cat)
            current_data, labels = data_load(path1, label=Label[i], usecols=usecols_cur, signal_size=signal_size, num_per_cat=num_per_cat)
            print('{} has been loaded'.format(path1))
            num_segments = len(vibration_data)
            sample_duration = signal_size / frequency  # seconds

            # 2. Read audio data
            root2 = os.path.join(root+'/SO_data', Category[i])
            datalist2 = os.listdir(root2)
            path2 = ''
            for fn in datalist2:
                if state1 in fn:
                    path2 = os.path.join(root2, fn)
            audio_data, _ = read_audio(path2, start_time, sample_duration, num_segments, Label[i])

            # 3. Combine multi-sensor and audio data
            for v, c, a, l in zip(vibration_data, current_data, audio_data, labels):
                data.append({'vibration': v, 'current': c, 'audio': a})
                lab.append(l)
    return [data, lab]

def read_audio(filename, start_time, duration, num_segments, label):
    """
    Read and segment audio data sample from a WAV file.
    Args:
        filename (str): WAV file path
        start_time (float): start time in seconds to begin reading
        duration (float): duration in seconds for each segment
        num_segments (int): number of segments to extract
        label (int): label for the audio segments
    Returns:
        list: Segmented audio data as a list of numpy arrays.
        list: Corresponding labels for each segment.
    """
    segments = []
    lab = []
    with wave.open(filename, 'rb') as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        nframes = wf.getnframes()
        total_duration = nframes / framerate

        frame_duration = int(duration * framerate)
        start_frame = int(start_time * framerate)

        for i in range(num_segments):
            pos = start_frame + i * frame_duration
            if pos + frame_duration > nframes:
                break
            wf.setpos(pos)
            str_data = wf.readframes(frame_duration)
            if sampwidth == 1:
                dtype = np.int8
            elif sampwidth == 2:
                dtype = np.int16
            elif sampwidth == 4:
                dtype = np.int32
            else:
                raise ValueError("Unsupported sample width")
            audio_data_buffer = np.frombuffer(str_data, dtype=dtype)
            num_samples_per_channel = len(audio_data_buffer) // n_channels
            audio_data_reshaped = audio_data_buffer.reshape((num_samples_per_channel, n_channels))
            # Convert to float32
            audio_data_reshaped = audio_data_reshaped.astype(np.float32)
            segments.append(audio_data_reshaped)
            lab.append(label)
    return segments, lab

#--------------------------------------------------------------------------------------------------------------------
class MOTOR_MultiSensor(object):
    def __init__(self, data_dir, condition, normlizetype="-1-1",):
        self.data_dir = data_dir
        self.condition = condition
        self.normlizetype = normlizetype
        self.data_transforms = {
            'train': {
                'vibration': Compose([
                    Reshape(),
                    # Normalize(self.normlizetype),
                    Retype(),
                    STFT_Transform(n_fft=127, hop_length=33),
                ]),
                'current': Compose([
                    Reshape(),
                    # Normalize(self.normlizetype),
                    Retype(),
                ]),
                'audio': Compose([
                    Reshape(),
                    Retype(),
                    Audio_Normalization(),
                    # Normalize(self.normlizetype),
                    MelSpectrogram_Transform(n_fft=512, hop_length=138, n_mels=64),
                ])
            },
            'val': {
                'vibration': Compose([
                    Reshape(),
                    # Normalize(self.normlizetype),
                    Retype(),
                    STFT_Transform(n_fft=127, hop_length=33),
                ]),
                'current': Compose([
                    Reshape(),
                    # Normalize(self.normlizetype),
                    Retype(),
                ]),
                'audio': Compose([
                    Reshape(),
                    Retype(),
                    Audio_Normalization(),
                    # Normalize(self.normlizetype),
                    MelSpectrogram_Transform(n_fft=512, hop_length=138, n_mels=64),
                ])
            }
        }

    def data_split(self, test_size):
        # get source train and val
        list_data = get_multimodal(self.data_dir, self.condition)
        samples = []
        for d, l in zip(list_data[0], list_data[1]):
            sample = d.copy()
            sample['label'] = l
            samples.append(sample)
        data_pd = pd.DataFrame(samples)
        train_pd, val_pd = train_test_split(data_pd, test_size=test_size, random_state=40, stratify=data_pd["label"])
        source_train = dataset(list_data=train_pd, transform=self.data_transforms['train'])
        source_val = dataset(list_data=val_pd, transform=self.data_transforms['val'])
        return source_train, source_val
    
    def data_return(self):
        list_data = get_multimodal(self.data_dir, self.condition)
        samples = []
        for d, l in zip(list_data[0], list_data[1]):
            sample = d.copy()
            sample['label'] = l
            samples.append(sample)
        data_pd = pd.DataFrame(samples)
        data_all = dataset(list_data=data_pd, transform=self.data_transforms['val'])
        return data_all