"""
Project: CreateDSforDML
Source: https://github.com/gubinmv/CreateDSforDML.git
Author: Maxim Gubin
"""

import librosa
import numpy as np
from numpy.lib import stride_tricks

import scipy
import scipy.io.wavfile
import soundfile as sf
import os, fnmatch

import tensorflow as tf

# класс хранит параметры для настройки
class args(object):

    home_dir    = './'

    # wave parameters
    fs          = 8000

    # spectrogramm parameters
    window      = 512
    k_overlap   = 3/4
    overlap     = int(k_overlap * window)
    step_wave   = int(window - overlap)

    # size of spectrogram
    img_rows    = 1 + window // 2
    img_cols    = 24

    # size wave from spectrogram
    maxRazmer = (window - overlap) * (img_cols - 1) + window

    # source DataSet
    path_source_DataSet     = './DataSet/dataset/'
    path_source_dataset_csv = './DataSet/dataset.csv'

    # dataset parameters
    class_noise                 = ['noise1']
    n_records_noise             = 100

    class_engine             = ['imbalance', 'vertical-misalignment', 'horizontal-misalignment', 'overhang_ball_fault', 'overhang_cage_fault', 'overhang_outer_race', 'underhang_ball_fault', 'underhang_outer_race']
    n_records_engine         = 130


    # ratio noise to voice
    k_mix_noise             = 0.9

    # dataset [x,y] in npz-files
    path_to_file_test_npz   = "./TestSet.npz"
    path_to_file_train_npz  = "./TrainSet.npz"

    # model trening
    model_name              = 'Forward'
    file_name_model         = 'model_conv1d-15skip'             #'model_conv1d-LSTM'
    file_history            = 'history_conv1d-15skip'
    path_history            = './' + file_history + '.txt'
    path_model              = './' + file_name_model + '.hdf5'

    # marking voice
    markup_voice_wav = 'train_data_y.wav'
    markup_file_npz = 'markup_train_voice.npz'
    markup_test_voice_wav = 'test_data_y.wav'
    markup_test_file_npz = 'markup_test_voice.npz'

    # dataset for classification
    classification_file_npz = 'Train_class.npz'
    classification_wav = 'train_data_x.wav'

    # params learning
    batch_size  = 64
    epochs      = 30

    # params CNN
    l2_lambda   = 1e-5

    # path of save parameters
    path_wav_out            = './!Out/'

def find_files(directory, pattern=['*.wav', '*.WAV']):
    '''find files in the directory'''
    files = []
    for root, dirnames, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, pattern[0]):
            files.append(os.path.join(root, filename))

    return files

# Конверт к 8кГц 16бит моно
def downsampling(source_files):
    '''Your new sampling rate'''

    new_rate = 8000

    for file in source_files:
            print("downsampling file = ", file)

            data, samplerate = sf.read(file)
            sf.write(file, data, samplerate, subtype='PCM_16')


            sampling_rate, audio = scipy.io.wavfile.read(file)
            if (len(audio.shape)==2):
                audio = audio.sum(axis=1) // 2
            number_of_samples = round(len(audio) * float(new_rate) / sampling_rate)
            audio = scipy.signal.resample(audio, number_of_samples)
            audio = audio.astype(dtype = np.int16)
            scipy.io.wavfile.write(file, new_rate, audio)

#выполняет нормализацию wav-файлов по списку файлов source_files
def norm_audio(source_files):
        '''Normalize the audio files before training'''

        for file in source_files:
            audio, sr = librosa.load(file, sr=8000)
##            sr, audio = scipy.io.wavfile.read(file)
            div_fac = 1 / np.max(np.abs(audio)) / 3.0
            audio = audio * div_fac
##            scipy.io.wavfile.write(file, sr, audio)
            sf.write(file, audio, sr)
            print("normalization file = ", file)


def remove_silent(audio):
    trimed_audio = []
    indices = librosa.effects.split(audio, hop_length=args.window, top_db=20)

    for index in indices:
        trimed_audio.extend(audio[index[0]: index[1]])
    return np.array(trimed_audio)

def get_wav_files(list_file):

    path_from = args.path_source_DataSet
    wave_data = []
    for i in range(len(list_file)):
        path_file = path_from + str(list_file['file_name'].values[i])
        wave_data_hlp, sr = librosa.load(path_file, sr=8000)
##        sr, wave_data_hlp = scipy.io.wavfile.read(path_file)
        wave_data_hlp = remove_silent(wave_data_hlp)
        wave_data.append(wave_data_hlp)

    return wave_data

#get stft
def my_stft(sig, frameSize=args.window, overlapFac=args.k_overlap, window=np.hanning): #256
    """ short time fourier transform of audio signal """

    win = window(frameSize)
    hopSize = int(frameSize - int(overlapFac * frameSize))
    samples = np.array(sig, dtype='float64')
    cols = int(np.ceil((len(samples) - frameSize) / float(hopSize)) + 1)
    samples = np.append(samples, np.zeros(frameSize))
    frames = stride_tricks.as_strided(
        samples,
        shape=(cols, frameSize),
        strides=(samples.strides[0] * hopSize, samples.strides[0])).copy()
    frames *= win

    return np.fft.rfft(frames)

def my_stft_2(wave_data, frameSize=args.window, overlapFac=args.overlap):
    """ short time fourier transform of audio signal """

    f, t, in_stft = scipy.signal.stft(wave_data, fs=args.fs, nperseg=frameSize, noverlap = overlapFac, window='hann')
    in_stft = np.transpose(in_stft)

    return in_stft

#Получаем спектрограммы из  wav-файлов
def get_stft_samples(wav_file):

    img_rows, img_cols = args.img_rows, args.img_cols

##    in_stft = my_stft(sig=wav_file)
    in_stft = my_stft_2(wave_data=wav_file)

    in_stft_amp = np.maximum(np.abs(in_stft*1000), 1e-5)
    in_data = in_stft_amp
    in_data = np.transpose(in_data)

    num_samples = in_data.shape[1]-img_cols

    sftt_frame = np.array([in_data[:, i:i+img_cols] for i in range(0, num_samples, 1)])

    return sftt_frame

def save_wav_file(file_name, wav_data):

    scipy.io.wavfile.write(file_name, 8000, wav_data)

def my_stoi(wav_clean, wav_predict):
    '''
         Calculate the STOI indicator between the two audios.
         The larger the value, the better the separation.
         input:
               wav_predict: Generated audio
               wav_clean:  Ground Truth audio
         output:
               SNR value
    '''

    k = 15 # one-third octave band
    num_band_in_octave = len(wav_clean) // k

    a = np.array([np.sqrt(sum(wav_clean[num_band_in_octave * i:num_band_in_octave + num_band_in_octave * i]**2)) for i in range(0, k, 1)])
    a = np.append(a, np.sqrt(sum(wav_clean[num_band_in_octave * k:]**2)))

    return a

def get_SDR(wav_clean, wav_predict, fs = args.fs, window = args.window, overlap = args.overlap):
    '''
         Calculate the SNR indicator between the two audios.
         The larger the value, the better the separation.
         input:
               wav_predict: Generated audio
               wav_clean:  Ground Truth audio
         output:
               SNR value
    '''

    f_1, t_1, ftt_clean_voice = scipy.signal.stft(wav_clean, fs=fs, nperseg=window, noverlap = overlap, window='hann')
    f_2, t_2, ftt_predict_voice = scipy.signal.stft(wav_predict, fs=fs, nperseg=window, noverlap = overlap, window='hann')

    ftt_clean_voice = np.transpose(ftt_clean_voice.real)
    ftt_predict_voice = np.transpose(ftt_predict_voice.real)

    num_segments = min(len(ftt_clean_voice),len(ftt_predict_voice))-10
    SDR_l2 = np.linalg.norm(ftt_clean_voice[:num_segments], ord=2)**2 / (np.linalg.norm((ftt_clean_voice[:num_segments] - ftt_predict_voice[:num_segments]), ord=2)**2)
    SDR_l2 = 10*np.log10(SDR_l2)

    return SDR_l2

def SI_SNR(wav_predict, wav_clean):
    '''
         Calculate the SI_SNR indicator between the two audios.
         The larger the value, the better the separation.
         input:
               wav_predict: Generated audio
               wav_clean:  Ground Truth audio
         output:
               SNR value
    '''
    wav_predict = wav_predict - np.mean(wav_predict)
    wav_clean = wav_clean - np.mean(wav_clean)
    s_target = sum(np.multiply(wav_predict, wav_clean))*wav_clean/np.power(np.linalg.norm(wav_clean, ord=2), 2)
    e_noise = wav_predict - s_target

    return 20*np.log10(np.linalg.norm(s_target, ord=2)/np.linalg.norm(e_noise, ord=2))

