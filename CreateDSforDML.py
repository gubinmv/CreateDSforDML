"""
Project: CreateDSforDML
Source: https://github.com/gubinmv/CreateDSforDML.git
Author: Maxim Gubin
"""
import os
import numpy as np
import pandas

from param_project import args
import param_project
from sklearn.utils import shuffle
from keras.utils import np_utils

fs = args.fs
window = args.window
overlap = args.overlap
step_wave = args.step_wave

img_rows, img_cols = args.img_rows, args.img_cols
maxRazmer = args.maxRazmer

def get_mix(noise, engine):

    len_noise = len(noise)
    len_engine = len(engine)

    if (len_engine > len_noise):
        k_ = 1 + len_engine // len_noise
        noise = [noise] * k_
        noise = np.concatenate(noise, axis=0)
    else:
        k_ = 1 + len_noise // len_engine
        engine = [engine] * k_
        engine = np.concatenate(engine, axis=0)

    tot_len = min(len(engine), len(noise))

    mix = engine[:tot_len] + noise[:tot_len]

    return engine[:tot_len], noise[:tot_len], mix


def get_data_set(list_noise, list_engine):

    print("\n Loading wav files ...")

    list_wav_data_noise = param_project.get_wav_files(list_noise)
    list_wav_data_engine = param_project.get_wav_files(list_engine)

    noise = np.concatenate(list_wav_data_noise, axis=0)
    engine = np.concatenate(list_wav_data_engine, axis=0)


    engine, noise, mix = get_mix(noise, engine)

    stft_x = param_project.get_stft_samples(wav_file=engine)

    x_train = np.array([])
    x_train = stft_x

    print('\n shape baze')
    print("x_train.shape = ",x_train[0].shape)
    print('len train records x = ', len(x_train))

    #reshape tensor from conv_1D_inverse
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols)

    return x_train

class_engine = args.class_engine
n_records_engine = args.n_records_engine

class_noise = args.class_noise
n_records_noise = args.n_records_noise

#open dataset
dataset_csv = args.path_source_dataset_csv
df = pandas.read_csv(dataset_csv, delimiter=',')
print(df['class'].value_counts())

len_train_spec = 30000

for i in range(len(class_engine)):

    #отобрать записи из датасета
    dataset_class = [class_noise[0]]
    n_records = args.n_records_noise
    list_csv_noise_train = df[df['class'].isin(dataset_class)].sample(n_records)[['file_name', 'class', 'length']]
    print("\n list noise")
    print(list_csv_noise_train.groupby('class').count())

    #
    dataset_class = [class_engine[i]]
    n_records = args.n_records_engine
    list_csv_engine_train = df[df['class'].isin(dataset_class)].sample(n_records)[['file_name', 'class', 'length']]
    print("\n list engine")
    print(list_csv_engine_train.groupby('class').count())

    x_data = get_data_set(list_csv_noise_train, list_csv_engine_train)
    len_train_spec = len(x_data) - len(x_data) // 10

    if (i==0):
        data_X_train = x_data[:len_train_spec]
        data_X_test = x_data[len_train_spec:]
    else:
        data_X_train = np.concatenate((data_X_train, x_data[:len_train_spec]), axis=0)
        data_X_test = np.concatenate((data_X_test, x_data[len_train_spec:]), axis=0)


    hlp = np_utils.to_categorical(i, len(class_engine))
    y_data = []
    y_data += [hlp for j in range(len(x_data))]
    y_data = np.array(y_data)
    if (i==0):
        data_Y_train = y_data[:len_train_spec]
        data_Y_test = y_data[len_train_spec:]
    else:
        data_Y_train = np.concatenate((data_Y_train, y_data[:len_train_spec]), axis=0)
        data_Y_test = np.concatenate((data_Y_test, y_data[len_train_spec:]), axis=0)

data_X_train = data_X_train.reshape(data_X_train.shape[0], img_rows, img_cols, 1)
data_X_test  = data_X_test.reshape(data_X_test.shape[0], img_rows, img_cols, 1)

print("save in npz-file ...")
path_to_file_npz = args.path_to_file_train_npz
print("\n total spectrogram train=", len(data_X_train))
print("\n total spectrogram test=", len(data_X_test))

np.savez(path_to_file_npz, DataX_train=data_X_train, DataY_train=data_Y_train, DataX_test=data_X_test, DataY_test=data_Y_test)
print("\n End programm creating spectrogramm.")
