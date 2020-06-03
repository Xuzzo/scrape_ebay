import tensorflow as tf
import os
import matplotlib.pyplot as plt
import random
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pickle
from sklearn import metrics
from PIL import Image


DATA_PATH = os.path.join('/Users/mmfp/Desktop', 'pokemon_cards_ds')
MODEL_PATH = os.path.join('/Users/mmfp/Desktop', 'ebay')
continue_train = False
is_train_model = True

EPOCHS = 200
TOT_CARDS = os.listdir(DATA_PATH)
TOT_CARDS.remove('.DS_Store')
if continue_train:
    with open(os.path.join(MODEL_PATH, 'tr_val_test.pkl'), 'rb') as f:
        CARDS_FOR_TRAIN, CARDS_FOR_VAL, CARDS_FOR_TEST, TOT_CARDS_NO_TEST = pickle.load(f)
else:
    CARDS_FOR_TEST = random.sample(TOT_CARDS, 2)
    TOT_CARDS_NO_TEST = [e for e in TOT_CARDS if e not in CARDS_FOR_TEST]
    CARDS_FOR_VAL = random.sample(TOT_CARDS_NO_TEST, 2)
    CARDS_FOR_TRAIN = [e for e in TOT_CARDS_NO_TEST if e not in CARDS_FOR_VAL]
    with open(os.path.join(MODEL_PATH, 'tr_val_test.pkl'), 'wb') as f:
            pickle.dump([CARDS_FOR_TRAIN, CARDS_FOR_VAL, CARDS_FOR_TEST, TOT_CARDS_NO_TEST], f)

def create_ds(data_len = 3000, ds_type = 'train'):
    if ds_type == 'train':
        dir_type = CARDS_FOR_TRAIN
    elif ds_type == 'val':
        dir_type = CARDS_FOR_VAL
    elif ds_type == 'test':
        dir_type = CARDS_FOR_TEST
    reference_img = []
    new_img = []
    label_list = []
    while len(reference_img) < data_len:
        is_match = bool(random.getrandbits(1))
        if is_match:
            card1_type = random.choice(dir_type)
            card1_dir = os.path.join(DATA_PATH, card1_type, 'true')
            card2_dir = card1_dir
            card1, card2 = random.sample(os.listdir(card1_dir), 2)
            label = 1
        else:
            card1_type = random.choice(dir_type)
            card1_dir = os.path.join(DATA_PATH, card1_type, 'true')
            is_internal = bool(random.getrandbits(1))
            card2_dir = os.path.join(DATA_PATH, card1_type, 'false')
            if (not is_internal) or len(os.listdir(card2_dir))==0:
                while True:
                    card2_type = random.choice(TOT_CARDS_NO_TEST)
                    if card2_type!= card1_type: break
                card2_dir = os.path.join(DATA_PATH, card2_type, 'true')
            card1 = random.choice(os.listdir(card1_dir))
            card2 = random.choice(os.listdir(card2_dir))
            label = 0
        if card1 == '.DS_Store':continue
        if card2 == '.DS_Store':continue
        image1 = tf.keras.preprocessing.image.load_img(os.path.join(card1_dir, card1),
                                                       color_mode="rgb",
                                                       target_size=(225, 169))
        image1 = keras.preprocessing.image.img_to_array(image1)
        image2 = tf.keras.preprocessing.image.load_img(os.path.join(card2_dir, card2),
                                                       color_mode="rgb",
                                                       target_size=(225, 169))
        image2 = keras.preprocessing.image.img_to_array(image2)
        reference_img.append(image1)
        new_img.append(image2)
        label_list.append(label)
    data = [np.array(reference_img), np.array(new_img), np.array(label_list)]
    return data


def define_model():
    card1_input = keras.Input(shape = (225, 169, 3), name = "reference_img")
    x = layers.Conv2D(32, 3, activation="relu")(card1_input)
    # x = layers.Conv2D(64, 3, activation="relu")(x)
    x = layers.MaxPooling2D(3)(x)
    # x = layers.Conv2D(32, 3, activation="relu")(x)
    cnn1_output = layers.Conv2D(32, 3, activation="relu")(x)
    # x = layers.MaxPooling2D(3)(x)
    # x = layers.Conv2D(32, 3, activation="relu")(x)
    # cnn1_output = layers.GlobalMaxPooling2D()(x)


    card2_input = keras.Input(shape = (225, 169, 3), name = "new_img")
    x = layers.Conv2D(32, 3, activation="relu")(card2_input)
    # x = layers.Conv2D(64, 3, activation="relu")(x)
    x = layers.MaxPooling2D(3)(x)
    # x = layers.Conv2D(32, 3, activation="relu")(x)
    cnn2_output = layers.Conv2D(32, 3, activation="relu")(x)
    # x = layers.MaxPooling2D(3)(x)
    # x = layers.Conv2D(32, 3, activation="relu")(x)
    # cnn2_output = layers.GlobalMaxPooling2D()(x)
    x = layers.concatenate([cnn1_output, cnn2_output])
    x = layers.Conv2D(64, 3, activation="relu")(x)
    x = layers.Conv2D(64, 3, activation="relu")(x)
    x = layers.MaxPooling2D(3)(x)
    x = layers.Flatten()(x)
    # x = layers.concatenate([cnn1_output, cnn2_output])
    # x = layers.Dense(32, activation = 'relu')(x)
    match_probability = layers.Dense(1, name="match", activation = 'sigmoid')(x)
    # match_probability = layers.Softmax(x)
    model = keras.Model(inputs = [card1_input, card2_input], outputs = [match_probability])
    model.compile(optimizer=keras.optimizers.RMSprop(1e-6),
                  loss={'match': keras.losses.BinaryCrossentropy(from_logits=False)},
                  loss_weights=[1.0])
    return model


def train_model():
    model = define_model()
    keras.utils.plot_model(model, 'pokemon_comparison_cnn.png', show_shapes = True)
    if continue_train:
        model.load_weights(os.path.join(MODEL_PATH, "model.h5"))
    ds = create_ds()
    ds_val = create_ds(data_len = 100, ds_type = 'val')
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience = 2,
         restore_best_weights=True)
    history = model.fit(
        {'reference_img': ds[0], 'new_img': ds[1]},
        {'match': ds[2]},
        epochs = EPOCHS,
        batch_size = 128,
        callbacks = [callback],
        validation_data=({'reference_img': ds_val[0], 'new_img': ds_val[1]},
                         {'match': ds_val[2]})
    )
    # serialize model to JSON
    model_json = model.to_json()
    with open(os.path.join(MODEL_PATH, "model.json"), "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(os.path.join(MODEL_PATH, "model.h5"))
    del ds, ds_val
    print("Saved model to disk")
    return model


def main():
    if is_train_model:
        model = train_model()
    else:
        json_file = open(os.path.join(MODEL_PATH, 'model.json'), 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = keras.models.model_from_json(loaded_model_json)
        # load weights into new model
        model.load_weights(os.path.join(MODEL_PATH, "model.h5"))
        print("Loaded model from disk")
    ds_test = create_ds(data_len = 500, ds_type = 'test')
    predictions = model.predict({'reference_img': ds_test[0], 'new_img': ds_test[1]}).flatten()
    preds = np.where(predictions<0.5, 0, 1)
    print(metrics.f1_score(ds_test[2], preds))
    for i in range(len(ds_test[0])):
        keras.preprocessing.image.array_to_img(ds_test[0][i]).show()
        keras.preprocessing.image.array_to_img(ds_test[1][i]).show()
        print(preds[i], ds_test[2][i])
        breakpoint()


if __name__=='__main__':
    main()
