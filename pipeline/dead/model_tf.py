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
from tensorflow.keras.preprocessing.image import ImageDataGenerator

assert tf.version.VERSION[0] == '2', "Must use TF Version 2.x"


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

    model.compile(optimizer='rmsprop',loss=tf.keras.losses.CategoricalCrossentropy(),
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