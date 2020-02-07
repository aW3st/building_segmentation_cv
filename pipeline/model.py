# ----------------------------- #
# Tensorflow image segmentation #
# ----------------------------- #

from __future__ import absolute_import, division, print_function, unicode_literals


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import pydot

import pdb

# Tensorflow stuff
import tensorflow as tf
from tqdm import tqdm

import sys, os


# !pip install -q git+https://github.com/tensorflow/examples.git
from tensorflow_examples.models.pix2pix import pix2pix

# import tensorflow_datasets as tfds
# tfds.disable_progress_bar()


assert tf.version.VERSION[0] == '2', "Must use TF Version 2.x"


def normalize(input_image, input_mask):
    input_image = tf.image.resize(input_image, (128, 128))
    input_mask = tf.image.resize(input_mask, (128, 128))
    input_image = tf.cast(input_image, tf.float32) / 255.0
    input_mask = tf.cast(input_mask, tf.float32)
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
    # print('image decoded type', type(image_decoded))


    # input_image, input_mask = tf.image.decode_jpeg(, tf.image.decode_jpeg(

    # image, mask = normalize(image_decoded, mask_decoded)

    # print(image.shape, mask.shape)


    return image_decoded, mask_decoded

# def data_gen(X=None, y=None, batch_size=32, nb_epochs=1, sess=None):
def generate_dataset_from_local(split='train'):
    
    # Create tensorflow dataset generator from directory of training examples:
    filepaths_ds = tf.data.Dataset.list_files('data/'+split+'/*[i]*')
    # print('sample file string tensor: ', next(iter(filepaths_ds)))
    labeled_ds = filepaths_ds.map(process_path)
    # print(type(labeled_ds))

    # # Test generator:
    # for f in train_ds.take(5):
    #     print(f.numpy())

    # Test proper read of image binaries
    # for image_raw, label_raw in dataset.take(1):
    #     print(repr(image_raw.numpy()[:100]))
    #     print()
    #     print(repr(label_raw.numpy()[:100]))

    return labeled_ds


# -–––––– From original tutorial. May not be needed. ––––––––––––
@tf.function
def load_image_train(datapoint):
    input_image = tf.image.resize(datapoint['image'], (512, 512))
    input_mask = tf.image.resize(datapoint['mask'], (512, 512))

    # Randomly flip the mask / image left-right.
    if tf.random.uniform(()) > 0.5:
        input_image = tf.image.flip_left_right(input_image)
    input_mask = tf.image.flip_left_right(input_mask)

    input_image, input_mask = normalize(input_image, input_mask)

    return input_image, input_mask


def load_image_test(datapoint):
    input_image = tf.image.resize(datapoint['image'], (128, 128))
    input_mask = tf.image.resize(datapoint['mask'], (128, 128))

    input_image, input_mask = normalize(input_image, input_mask)

    return input_image, input_mask
# –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––


def partition_datasets(dataset=None, val_set=False):
    ''''
    Given dictionary of tf.data.datasets (keys are 'train', 'test'[, 'val']),

    return training, test, and optionally validation sets.
    '''

    # need to implement a proper test/train/val split at some point
    if val_set:
        raise NotImplementedError
    else:
        train = generate_dataset_from_local('train')
        test = generate_dataset_from_local('test')

        # From original tutorial:
        # train = dataset['train'].map(load_image_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        # test = dataset['test'].map(load_image_test)

    # Buffer and epoch sizes
    TRAIN_LENGTH = len(list(train.enumerate()))
    BATCH_SIZE = 64
    BUFFER_SIZE = 1000
    STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE
    OUTPUT_CHANNELS = 2

    train_dataset = train.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
    train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    test_dataset = test.batch(BATCH_SIZE)

    train, test, train_dataset, test_dataset


def compile_model(architecture='unet', pretrained_encoder='mobilenetv2'):

    def build_unet(OUTPUT_CHANNELS=2):
        '''
        Build UNet model.
        '''

        # This is the last layer of the model
        last = tf.keras.layers.Conv2DTranspose(
            OUTPUT_CHANNELS, 3, strides=2,
            padding='same', activation='softmax')  #64x64 -> 128x128

        inputs = tf.keras.layers.Input(shape=[128, 128, 3])
        x = inputs

        # Downsampling through the model
        skips = encoder_layers(x)
        x = skips[-1]
        skips = reversed(skips[:-1])

        # Upsampling and establishing the skip connections
        for decoder, skip in zip(decoder_layers, skips):
            x = decoder(x)
            concat = tf.keras.layers.Concatenate()
            x = concat([x, skip])

        x = last(x)

        return tf.keras.Model(inputs=inputs, outputs=x)

    if pretrained_encoder == 'mobilenetv2':
        # Pretrained Encoder
        base_model = tf.keras.applications.MobileNetV2(input_shape=[128, 128, 3], include_top=False)

        # Use the activations of these layers
        layer_names = [
            'block_1_expand_relu',   # 64x64
            'block_3_expand_relu',   # 32x32
            'block_6_expand_relu',   # 16x16
            'block_13_expand_relu',  # 8x8
            'block_16_project',      # 4x4
        ]
        layers = [base_model.get_layer(name).output for name in layer_names]

        # Create the feature extraction model
        encoder_layers = tf.keras.Model(inputs=base_model.input, outputs=layers)
        encoder_layers.trainable = False
        
    else:
        raise NotImplementedError

    # Pix2pix architecture abstracts various convolutions and upsampling methods
    decoder_layers = ([
        pix2pix.upsample(512, 3),  # 4x4 -> 8x8
        pix2pix.upsample(256, 3),  # 8x8 -> 16x16
        pix2pix.upsample(128, 3),  # 16x16 -> 32x32
        pix2pix.upsample(64, 3),   # 32x32 -> 64x64
        ])

    if architecture=='unet':
        model = build_unet(OUTPUT_CHANNELS=2)
    else:
        raise NotImplementedError

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

    return model


# Evaluation utilities
def create_mask(pred_mask):
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask[0]

def show_predictions(dataset=None, num=1):
  if dataset:
    for image, mask in dataset.take(num):
        pred_mask = model.predict(image)
        display([image[0], mask[0], create_mask(pred_mask)])
  else:
    display([sample_image, sample_mask,
            create_mask(model.predict(sample_image[tf.newaxis, ...]))])

# Callback function for training
class DisplayCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs=None):
    clear_output(wait=True)
    show_predictions()
    print ('\nSample Prediction after epoch {}\n'.format(epoch+1))

def train(model, train_dataset, validation_dataset):
    '''
    Train model. Return history.
    '''

    EPOCHS = 20
    VAL_SUBSPLITS = 5
    VAL_LENGTH = 1414
    VALIDATION_STEPS = (VAL_LENGTH // BATCH_SIZE )// VAL_SUBSPLITS

    model_history = model.fit(train_dataset, epochs=EPOCHS,
                            steps_per_epoch=STEPS_PER_EPOCH,
                            validation_steps=VALIDATION_STEPS,
                            validation_data=validation_dataset,
                            callbacks=[DisplayCallback()])

    # return model_history
    return model_history


def run_model_workflow():

    train = generate_dataset_from_local('train')
    print(train)
    train = train.map(normalize, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    print(train)

    test = generate_dataset_from_local('test')
    test.map(normalize)

    dataset = {'train': train, 'test': test}

        # From original tutorial:
        # train = dataset['train'].map(load_image_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        # test = dataset['test'].map(load_image_test)

    # Buffer and epoch sizes
    TRAIN_LENGTH = 1414
    BATCH_SIZE = 64
    BUFFER_SIZE = 300
    STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE
    OUTPUT_CHANNELS = 2

    train_dataset = train.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
    train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    print(train_dataset)
    test_dataset = test.batch(BATCH_SIZE)

    # Compile model

    model = compile_model()
    # Train

    # tf.keras.utils.plot_model(model, show_shapes=True)

    # pdb.set_trace()

    # model_history = model.fit(train_dataset, epochs=20, steps_per_epoch=STEPS_PER_EPOCH)

    # Return model history and results.
    pdb.set_trace()
    train(model, train_dataset, test_dataset)
    pass