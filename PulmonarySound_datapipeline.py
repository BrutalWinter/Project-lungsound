import struct
import os
import tensorflow as tf
from matplotlib import pyplot as plt
from PulmonarySound_FuncBase import Calculating_MFCCs_from_wave
###############################################
def plt_wav(wave_data, id):
    fig = plt.figure(figsize=(18, 4))
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.plot(wave_data, color='b')
    plt.title("wav data: %s" % id)
    plt.xlabel("Time(s)")
    plt.ylabel("Amplitude")
    plt.show()


# 读取label数据
def view_label(label_file):
    piece_len = 12
    data_handler = open(label_file, 'rb')
    file_len = os.path.getsize(label_file)
    label_list = []
    # print('file_len', file_len)
    for i in range(file_len // piece_len):
        # if i > 2:
        #     break
        data_handler.seek(i * piece_len)
        data = data_handler.read(piece_len)
        id, = struct.unpack('<I', data[:4])
        labels = struct.unpack('<' + 'I' * 2, data[4:])
        label_list.append(labels)

        # print('id', id)
        # print('labels', labels)
    data_handler.close()
    return label_list

#读取信号数据
def view_data(data_file):
    piece_len = 640004
    data_handler = open(data_file, 'rb')
    file_len = os.path.getsize(data_file)
    # print('file_len', file_len)
    wave_data_list=[]
    for i in range(file_len//piece_len):
        # if i >2:
        #     break
        data_handler.seek(i*piece_len)
        data = data_handler.read(piece_len)
        id, = struct.unpack('<I', data[:4])
        wave_data = list(struct.unpack('<' + 'f' * 160000, data[4:640004]))
        wave_data_list.append(wave_data)

        # print('id', id)
        # plt_wav(wave_data, id)
    data_handler.close()
    return  wave_data_list

##########################################
##########################################
def Parsed_LungSound_label(data):

    raw_data_efid= tf.strings.substr(data, 0, 4)
    efid = tf.io.decode_raw(raw_data_efid, tf.int32)  ## corresponding I

    raw_label = tf.strings.substr(data, 4 + 4 * 0, 4 * 2)  ######## number of heartbeat
    label = tf.io.decode_raw(raw_label, tf.int32)  ## corresponding B

    # ECG_extend_dim = tf.expand_dims(raw_data_Ecg , -1)

    return efid, label

def Parsed_LungSound_data(data):

    raw_data_efid= tf.strings.substr(data, 0, 4)
    efid = tf.io.decode_raw(raw_data_efid, tf.int32)  ## corresponding I

    raw_data = tf.strings.substr(data, 4 + 4 * 0, 4 * 160000)  ######## number of heartbeat
    data = tf.io.decode_raw(raw_data, tf.float32)  ## corresponding B

    return efid, data
##########################################
##########################################




if __name__ == '__main__':
    data_file = r'/home/brutal/PycharmProjects/Data_base/lung_sound_merge_label20210107/wave_data.DAT'
    label_file = r'/home/brutal/PycharmProjects/Data_base/lung_sound_merge_label20210107/label.DAT'

    ###########  ZN pipeline:
    # wave_list=view_data(data_file)
    # label_list=view_label(label_file)
    # print(wave_list)
    # print(label_list)

    ############    tf pipeline:
    Label_dataset = tf.data.FixedLengthRecordDataset(filenames = label_file, record_bytes = 12)
    Labels_dataset_parsed = Label_dataset.map(Parsed_LungSound_label,num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # Labels_Rpos_dataset_parsed = Label_Rpos_dataset.map(Parsed_heart_beat)

    Dataset = tf.data.FixedLengthRecordDataset(filenames=data_file, record_bytes=640004)
    Dataset_parsed = Dataset.map(Parsed_LungSound_data,num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # Ecg_dataset_parsed = Ecg_dataset.map(Parsed_Ecg_data)
    Label_Data_dataset=tf.data.Dataset.zip((Labels_dataset_parsed,Dataset_parsed)).batch(1)
    ##########################################
    start = 0
    end = 1

    for step, data in enumerate(Label_Data_dataset):
        if step > end + 1:
            break

        if step >= start and step <= end:
            print('==>The {:d}th -- its label is {}, shape={}, dtype={}'.format(step, data[0][1],data[0][1].shape,data[0][1].dtype))
            print('==>The {:d}th -- its data is {}, shape={}, dtype={}'.format(step, data[1][1].shape,data[1][1].shape,data[1][1].dtype))
            # print('label difference={}'.format(data[0][1]-label_list[step]))
            # print('data difference={}'.format(tf.math.reduce_sum(data[1][1] - wave_list[step])))

            PCM_batch=data[1][1]
            mfccs_batch=Calculating_MFCCs_from_wave(PCM_batch,sample_rate=8000)














