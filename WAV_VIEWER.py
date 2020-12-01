import wave
import matplotlib.pyplot as plt
import numpy as np
import os


def read_wave_data(file_path):
    # f = wave.open(file_path, "rb")
    with wave.open(file_path, "rb") as f:
        params = f.getparams()
        nchannels, sampwidth, framerate, nframes = params[:4]
        str_data = f.readframes(nframes)
        wave_data = np.frombuffer(str_data, dtype=np.short)
        wave_data = wave_data.T
        # calculate the time bar
        time = np.arange(0, nframes) * (1.0 / framerate)
        return wave_data, time, framerate, nframes








if __name__ == "__main__":
    wave_data, time, framerate, nframes = read_wave_data(r"/home/brutal/PycharmProjects/Digital Signal Process/HearSoundDataBase/a/artifact__201106040933.wav")
    print(nframes)
    print(framerate)
    fig1 = plt.figure(figsize=(25, 10))
    axes1 = fig1.add_subplot(1, 1, 1)
    axes1.plot(time, wave_data)
    plt.show()