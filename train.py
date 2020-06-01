#!/usr/bin/env python

import os.path
import sys

import math

from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras.optimizers import SGD

from callbacks import LearningRateScheduler
from losses import mean_corner_error
from models import create_model


def main():

    model = create_model()

    # Configuration
    batch_size = 64
    target_iterations = 90000 # at batch_size = 64
    base_lr = 0.005

    sgd = SGD(lr=base_lr, momentum=0.9)

    model.compile(optimizer=sgd, loss='mean_squared_error', metrics=[mean_corner_error])
    model.summary()

    save_path = os.path.dirname(os.path.realpath(__file__))
    checkpoint = ModelCheckpoint(os.path.join(save_path, 'model.{epoch:02d}.h5'))

    # LR scaling as described in the paper: https://arxiv.org/pdf/1606.03798.pdf
    lr_scheduler = LearningRateScheduler(base_lr, 0.1, 30000)

    # In the paper, the 90,000 iterations was for batch_size = 64
    # So scale appropriately
    target_iterations = int(target_iterations * 64 / batch_size)

    # load data here from ME3
    # separate into test and sample
    # play with the split, but you can start with:
    # Trainig: 80%
    # Testing: 20%

    TRAIN_SAMPLES_COUNT = 100
    TEST_SAMPLES_COUNT = 20

    # As stated in Keras docs
    steps_per_epoch = int(TRAIN_SAMPLES_COUNT / batch_size)
    epochs = int(math.ceil(target_iterations / steps_per_epoch))

    train_data = #load train data here
    test_data = #load test data here
    test_steps = int(TEST_SAMPLES_COUNT / batch_size)

    # Train
    model.fit_generator(loader, steps_per_epoch, epochs,
                        callbacks=[lr_scheduler, checkpoint],
                        validation_data=test_data, validation_steps=test_steps)

    # Step 1, make the training script work :)
    # Step 2, let me know if you get it working, the next for on the path will be open to you.
    # Note: Step 2 won't be discussed during this call.

if __name__ == '__main__':
    main()
