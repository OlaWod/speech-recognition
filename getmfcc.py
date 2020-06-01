import os
import librosa
import numpy as np


label_path = 'audio'
DATA = 'data.npy'
TARGET = 'target.npy'


# 加载标签
def load_label(label_path):
    label = os.listdir(label_path)
    return label


# 提取 mfcc 参数
def wav2mfcc(path, max_pad_size=11):
    y, sr = librosa.load(path=path, sr=None)
    audio_mac = librosa.feature.mfcc(y=y, sr=sr)
    y_shape = audio_mac.shape[1]
    if y_shape < max_pad_size:
        extra_pad_size = max_pad_size - y_shape
        audio_mac = np.pad(audio_mac, ((0, 0), (0, extra_pad_size)))
    else:
        audio_mac = audio_mac[:, :max_pad_size]
    return audio_mac


# 存储处理过的数据，方便下一次的使用
def save_data_to_array(label_path, max_pad_size=11):
    mfcc_vectors = []
    target = []
    labels = load_label(label_path=label_path)
    for i, label in enumerate(labels):
        path = label_path + '/' + label
        wavfiles = [path + '/' + file for file in os.listdir(path)]
        for wavfile in wavfiles:
            wav = wav2mfcc(wavfile, max_pad_size=max_pad_size)
            mfcc_vectors.append(wav)
            target.append(i)
        print(i)
    np.save(DATA, mfcc_vectors)
    np.save(TARGET, target)
    # return mfcc_vectors, target


def save():
    label_path = 'audio'
    save_data_to_array(label_path, max_pad_size=11)


if __name__ == '__main__':
    save()
