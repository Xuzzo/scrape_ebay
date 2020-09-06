import tensorflow as tf
import os
import matplotlib.pyplot as plt
import random
import numpy as np
import pickle
from sklearn import metrics
from PIL import Image
import keras
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Activation, Concatenate
from keras import backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import Model
from keras.layers import Input, Flatten, Dense
from train_utils import TrainUtils
from siamese import SiameseNetwork

EPOCHS = 1000

data_path = os.path.join('/Users/mmfp/Desktop', 'pokemon_cards_ds')
model_path = os.path.join('/Users/mmfp/Desktop', 'ebay_scrape')
batch_size = 100
num_classes = 2
epochs = 999999

# input image dimensions
input_shape = (225, 225)
tutils = TrainUtils(data_path, model_path)
x1_train, x2_train, y_train = tutils.create_ds(ds_type='train', input_shape = input_shape)
x1_val, x2_val, y_val = tutils.create_ds(ds_type='val', input_shape = input_shape)
x1_test, x2_test, y_test = tutils.create_ds(ds_type='test', input_shape = input_shape)

def create_base_model(input_shape):
    model_input = Input(shape=input_shape)
    embedding = Flatten()(model_input)
    embedding = Dense(128)(embedding)
    return Model(model_input, embedding)


def create_head_model(embedding_shape):
    embedding_a = Input(shape=embedding_shape)
    embedding_b = Input(shape=embedding_shape)
    
    head = Concatenate()([embedding_a, embedding_b])
    head = Dense(4)(head)
    head = BatchNormalization()(head)
    head = Activation(activation='sigmoid')(head)

    head = Dense(1)(head)
    head = BatchNormalization()(head)
    head = Activation(activation='sigmoid')(head)

    return Model([embedding_a, embedding_b], head)


base_model = create_base_model(input_shape)
head_model = create_head_model(base_model.output_shape)
siamese_network = SiameseNetwork(base_model, head_model)
siamese_network.compile(loss='binary_crossentropy', optimizer=keras.optimizers.adam(), metrics=['accuracy'])

siamese_checkpoint_path = "./siamese_checkpoint"

siamese_callbacks = [
    EarlyStopping(monitor='val_acc', patience=10, verbose=0),
    ModelCheckpoint(siamese_checkpoint_path, monitor='val_acc', save_best_only=True, verbose=0)
]

siamese_network.fit(x_train, y_train,
                    validation_data=(x_test, y_test),
                    batch_size=1000,
                    epochs=epochs,
                    callbacks=siamese_callbacks)

siamese_network.load_weights(siamese_checkpoint_path)
embedding = base_model.outputs[-1]
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# Add softmax layer to the pre-trained embedding network
embedding = Dense(num_classes)(embedding)
embedding = BatchNormalization()(embedding)
embedding = Activation(activation='sigmoid')(embedding)

model = Model(base_model.inputs[0], embedding)
model.compile(loss=keras.losses.binary_crossentropy,
              optimizer=keras.optimizers.adam(),
              metrics=['accuracy'])

model_checkpoint_path = "./model_checkpoint"

model__callbacks = [
    EarlyStopping(monitor='val_acc', patience=10, verbose=0),
    ModelCheckpoint(model_checkpoint_path, monitor='val_acc', save_best_only=True, verbose=0)
]

model.fit(x_train, y_train,
          batch_size=128,
          epochs=epochs,
          callbacks=model__callbacks,
          validation_data=(x_test, y_test))

model.load_weights(model_checkpoint_path)
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])