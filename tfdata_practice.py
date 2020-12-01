import numpy as np
import pandas as pd
import tensorflow as tf
import IPython.display as display

import struct
import os
import matplotlib.pyplot as plt
import wave
import csv
import sys
import soundfile as sf
import glob

def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def serialize_example(feature0, feature1, feature2, feature3):
  """
  Creates a tf.Example message ready to be written to a file.
  """
  # Create a dictionary mapping the feature name to the tf.Example-compatible
  # data type.
  feature = {
      'feature0': _int64_feature(feature0),
      'feature1': _int64_feature(feature1),
      'feature2': _bytes_feature(feature2),
      'feature3': _float_feature(feature3),
  }

  # Create a Features message using tf.train.Example.
  example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
  return example_proto.SerializeToString()
#####################################################################################################
#####################################################################################################


# The number of observations in the dataset.
n_observations = int(1e4)
# Boolean feature, encoded as False or True.
feature0 = np.random.choice([False, True], n_observations)
# Integer feature, random from 0 to 4.
feature1 = np.random.randint(0, 5, n_observations)
# String feature
strings = np.array([b'cat', b'dog', b'chicken', b'horse', b'goat'])
feature2 = strings[feature1]
# Float feature, from a standard normal distribution
feature3 = np.random.randn(n_observations)

serialized_example = serialize_example(False, 4, b'goat', 0.9876)
print(serialized_example)
example_proto = tf.train.Example.FromString(serialized_example)
print(example_proto)


# 若应用于数组的元组，将返回元组的数据集：
features_dataset = tf.data.Dataset.from_tensor_slices((feature0, feature1, feature2, feature3))
print(features_dataset)
for f0,f1,f2,f3 in features_dataset.take(1):
  print(f0)
  print(f1)
  print(f2)
  print(f3)

# ×该函数只能在graph模式下运行，也就是说必须在graph中定义并且返回tf.Tensors类型。
# 上面定义的serialize_example并不是返回tensor的函数，需要使用tf.py_function函数包装使其可以兼容，于是将上面定义的函数进行改进。
# 使用tf.py_dunction需要制定shape和type：
def tf_serialize_example(f0,f1,f2,f3): #Wraps a python function into a TensorFlow op that executes it eagerly.
  tf_string = tf.py_function(serialize_example,(f0,f1,f2,f3), tf.string)      # pass these args to the above function.# the return type is `tf.string`.
  return tf.reshape(tf_string, ()) # The return result is a scalar(one tensor)
print(tf_serialize_example(f0,f1,f2,f3))


# 将此函数应用于数据集中的每个元素：
serialized_features_dataset = features_dataset.map(tf_serialize_example)
print(serialized_features_dataset)

def generator():
  for features in features_dataset:
    yield serialize_example(*features)
# Creates a Dataset whose elements are generated by generator.
serialized_features_dataset = tf.data.Dataset.from_generator(generator, output_types=tf.string, output_shapes=())

# 将上面的结果写入TFRecord
filename = "training_1/test.tfrecord"
writer = tf.data.experimental.TFRecordWriter(filename)
writer.write(serialized_features_dataset)
print('done')


# 读取 TFRecord 文件
# 您还可以使用 tf.data.TFRecordDataset 类来读取 TFRecord 文件。
# 有关通过 tf.data 使用 TFRecord 文件的详细信息，请参见此处。使用 TFRecordDataset 对于标准化输入数据和优化性能十分有用。
filenames = [filename]
raw_dataset = tf.data.TFRecordDataset(filenames)
print(raw_dataset)
for raw_record in raw_dataset.take(10):
  print(repr(raw_record))


for raw_record in raw_dataset.take(1):
  example = tf.train.Example()
  example.ParseFromString(raw_record.numpy())
  print(example)


feature_description = {
    'feature0': tf.io.FixedLenFeature([], tf.int64, default_value=0),
    'feature1': tf.io.FixedLenFeature([], tf.int64, default_value=0),
    'feature2': tf.io.FixedLenFeature([], tf.string, default_value=''),
    'feature3': tf.io.FixedLenFeature([], tf.float32, default_value=0.0),
}

def _parse_function(example_proto):
  # Parse the input `tf.Example` proto using the dictionary above.
  return tf.io.parse_single_example(example_proto, feature_description)
parsed_dataset = raw_dataset.map(_parse_function)
print(parsed_dataset)
for parsed_record in parsed_dataset.take(10):
  print(repr(parsed_record))
#####################################################################################################
#####################################################################################################
# cat_in_snow  = tf.keras.utils.get_file('320px-Felis_catus-cat_on_snow.jpg', 'https://storage.googleapis.com/download.tensorflow.org/example_images/320px-Felis_catus-cat_on_snow.jpg')
# williamsburg_bridge = tf.keras.utils.get_file('194px-New_East_River_Bridge_from_Brooklyn_det.4a09796u.jpg','https://storage.googleapis.com/download.tensorflow.org/example_images/194px-New_East_River_Bridge_from_Brooklyn_det.4a09796u.jpg')
#
# image_labels = {
#     cat_in_snow : 0,
#     williamsburg_bridge : 1,
# }
#
# image_string = open(cat_in_snow, 'rb').read()
#
# label = image_labels[cat_in_snow]
#
# # Create a dictionary with features that may be relevant.
# def image_example(image_string, label):
#   image_shape = tf.image.decode_jpeg(image_string).shape
#
#   feature = {
#       'height': _int64_feature(image_shape[0]),
#       'width': _int64_feature(image_shape[1]),
#       'depth': _int64_feature(image_shape[2]),
#       'label': _int64_feature(label),
#       'image_raw': _bytes_feature(image_string),
#   }
#
#   return tf.train.Example(features=tf.train.Features(feature=feature))
#
# for line in str(image_example(image_string, label)).split('\n')[:15]:
#   print(line)
# print('...')
#
# record_file = 'training_1/images.tfrecords'
# with tf.io.TFRecordWriter(record_file) as writer:
#   for filename, label in image_labels.items():
#     image_string = open(filename, 'rb').read()
#     tf_example = image_example(image_string, label)
#     writer.write(tf_example.SerializeToString())

#####################################################################################################
#####################################################################################################














