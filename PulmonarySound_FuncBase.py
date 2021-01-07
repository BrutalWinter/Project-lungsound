import os
import tensorflow as tf
import numpy as np
import soundfile as sf #pysoundfile 全部自动转换成float32来输出 numpy result
import matplotlib.pyplot as plt


def wave_resampled(wave_data, Fs_original, Fs_desire):
    time_total = len(wave_data) // Fs_original
    x_before= np.linspace(0, time_total, len(wave_data))
    x_after = np.linspace(0, time_total, time_total * Fs_desire)
    wave_data_resampled = np.interp(x_after, x_before, wave_data)
    return wave_data_resampled




#梅尔倒谱系数（mfcc）: 提取过程：连续语音--预加重--加窗分帧--FFT--MEL滤波器组--对数运算--DCT
def Calculating_MFCCs_from_wave(wave_data, sample_rate):
    # A 1024-point STFT with frames of 64 ms and 75% overlap.
    # wave_data=tf.cast(wave_data, dtype=tf.float32)
    frame_length,frame_step,fft_length = 1024, 256, 1024
    stfts = tf.signal.stft(wave_data, frame_length=frame_length, frame_step=frame_step, fft_length=fft_length, window_fn=tf.signal.hamming_window,
                           pad_end=True)
    spectrograms = tf.math.abs(stfts)

    # Warp the linear scale spectrograms into the mel-scale.
    print('The shape of STFTs={}'.format(spectrograms.shape))
    num_spectrogram_bins = stfts.shape[-1]
    # print(num_spectrogram_bins) # where fft_unique_bins is fft_length // 2 + 1 (the unique components of the FFT).

    nfilt = 40
    lower_edge_hertz, upper_edge_hertz, num_mel_bins = 0.0, sample_rate / 2, nfilt
    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(num_mel_bins, num_spectrogram_bins, sample_rate,
                                                                        lower_edge_hertz, upper_edge_hertz)
    print('filter bank shape={}'.format(linear_to_mel_weight_matrix.shape))
    ########## show the filter banks:##########
    # plot_FilterBanks(linear_to_mel_weight_matrix)
    # plt.show()
    #################### ##########
    linear_to_mel_weight_matrix=tf.cast(linear_to_mel_weight_matrix,dtype=tf.float32)
    mel_spectrograms = tf.tensordot(spectrograms, linear_to_mel_weight_matrix, 1)
    print('The shape of mel_spectrograms={}'.format(mel_spectrograms.shape))

    mel_spectrograms.set_shape(spectrograms.shape[:-1].concatenate(linear_to_mel_weight_matrix.shape[-1:]))
    # print(spectrograms.shape[:-1])
    # print(linear_to_mel_weight_matrix.shape[-1:])
    # print('The shape of mel_spectrograms after set={}'.format(mel_spectrograms.shape))

    # Compute a stabilized mel-scale spectrograms.
    mel_spectrograms = tf.where(mel_spectrograms <= 0, np.finfo(float).eps, mel_spectrograms)
    # A=np.array([[1,-1],[-2.,3],[4,5]])
    # print(A)
    # A=tf.where(A<=0,np.finfo(float).eps,A)
    # print(A)
    # log to get log-magnitude
    log_mel_spectrograms = tf.math.log(mel_spectrograms)  # .e-base
    # Must be one of the following types: bfloat16, half, float32, float64, complex64, complex128.
    print('The shape of log_mel_spectrograms={}'.format(log_mel_spectrograms.shape))
    #
    # Compute MFCCs from log_mel_spectrograms and take the first 13.
    Partial = 14
    mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrograms)[..., 0:Partial]
    # Mean Normalization:
    # mean_mfccs=tf.math.reduce_mean(mfccs, axis=1)
    # mean_mfccs = np.expand_dims(mean_mfccs, axis=1)
    # mean_mfccs_tiles=tf.tile(mean_mfccs,[1,mfccs.shape[1],1])
    # mfccs -= mean_mfccs_tiles

    print('The shape of mfccs={}'.format(mfccs.shape))


    return mfccs














if __name__ == "__main__":
    path=r'/home/brutal/PycharmProjects/Data_base/dataset-lunsound/ICBHI_final_database/101_1b1_Al_sc_Meditron.wav'
    sample_data, sample_rate = sf.read(path)
    print(sample_data)
    print(sample_rate)
    print(len(sample_data))
    fig1 = plt.figure(figsize=(25, 10))
    axes1 = fig1.add_subplot(1, 1, 1)
    axes1.plot(sample_data)
    plt.show()

    # audio_binary = tf.io.read_file(path)
    # audio_data=tf.audio.decode_wav(audio_binary)
    # print(audio_data)


    Audio_Directory = r'/home/brutal/PycharmProjects/Data_base/dataset-lunsound/ICBHI_final_database'
    Audio_files_path = os.path.join(Audio_Directory, '*.wav')
    Audio_files = sorted(tf.io.gfile.glob(Audio_files_path))
    # print(Audio_files)
    print('Audio_files:')
    for i in Audio_files:
        # print(i)
        sample_datas, sample_rates = sf.read(i)
        print(sample_datas.shape)
        # print(sample_rates)