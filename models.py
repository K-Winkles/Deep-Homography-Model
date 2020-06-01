#!/usr/bin/env python
# Darwin Bautista
# HomographyNet, from https://arxiv.org/pdf/1606.03798.pdf

import os.path

from keras.applications import MobileNet
from keras.models import Sequential, Model
from keras.layers import Conv2D, Dense, MaxPooling2D, InputLayer, Dropout, \
    BatchNormalization, Flatten, Concatenate

def create_model():
    model = Sequential(name='homographynet')
    model.add(InputLayer((128, 128, 2), name='input_1'))

    # 4 Layers with 64 filters, then another 4 with 128 filters
    filters = 4 * [64] + 4 * [128]
    for i, f in enum erate(filters, 1):
        model.add(Conv2D(f, 3, padding='same', activation='relu', name='conv2d_{}'.format(i)))
        model.add(BatchNormalization(name='batch_normalization_{}'.format(i)))
        # MaxPooling after every 2 Conv layers except the last one
        if i % 2 == 0 and i != 8:
            model.add(MaxPooling2D(strides=(2, 2), name='max_pooling2d_{}'.format(int(i/2))))

    model.add(Flatten(name='flatten_1'))
    model.add(Dropout(0.5, name='dropout_1'))
    model.add(Dense(1024, activation='relu', name='dense_1'))
    model.add(Dropout(0.5, name='dropout_2'))

    # Regression model
    model.add(Dense(8, name='dense_2'))

    return model