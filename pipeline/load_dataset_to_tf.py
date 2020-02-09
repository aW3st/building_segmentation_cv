# Load dataset from directory.

import os
import sys

import tensorflow as tf

tf.data.Dataset()

def normalize(input_image, input_mask):
    input_image = tf.image.resize(input_image, (128, 128))
    input_mask = tf.image.resize(input_mask, (128, 128))
    input_image = tf.cast(input_image, tf.float32) / 255.0
    input_mask = tf.cast(input_mask, tf.float32)
    print(input_image.shape, input_mask.shape)
    # input_mask -= 1
    return input_image, input_mask

def process_path(image_path):
    # trim 'i.jpg' from path and replace with 'mask.jpg'
    mask_path = tf.strings.regex_replace(image_path, '\1_(i).jpg', 'mask')
    
    # This will return a tuple of input & mask as the dataset format requires
    image_string = tf.io.read_file(image_path)
    mask_string = tf.io.read_file(mask_path)

    image_decoded = tf.image.decode_jpeg(image_string, channels=3)
    mask_decoded = tf.image.decode_jpeg(mask_string, channels=1)
    # print('image decoded type', type(image_decoded)

    return image_decoded, mask_decoded

# def data_gen(X=None, y=None, batch_size=32, nb_epochs=1, sess=None):
def load_dataset(directory='data/train'):

    image_file_pattern = directory + '/images/*.jpg'
    mask_file_pattern = directory + '/masks/*.jpg'

    tf.data.Dataset()

    image_ds = tf.data.Dataset.list_files(image_file_pattern, shuffle=False)
    mask_ds = tf.data.Dataset.list_files(mask_file_pattern, shuffle=False)

    zip_ds = tf.data.Dataset.zip((image_ds, mask_ds))
