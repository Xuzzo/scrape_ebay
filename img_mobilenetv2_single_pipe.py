import tensorflow as tf
import os
import matplotlib.pyplot as plt
import random
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
import pickle
from sklearn import metrics
from PIL import Image


DATA_PATH = os.path.join('/Users/mmfp/Desktop', 'pokemon_cards_ds')
MODEL_PATH = os.path.join('/Users/mmfp/Desktop', 'ebay')
CACHE_PATH = os.path.join('/Users/mmfp/Desktop', 'ebay', 'mobilenet_cache')
is_create_ds = False
base_model = MobileNetV2(include_top = False)

EPOCHS = 1000
TOT_CARDS = os.listdir(DATA_PATH)
TOT_CARDS.remove('.DS_Store')
CARDS_FOR_TEST = ['dark+blastoise', 'zacian+gold']
TOT_CARDS_NO_TEST = [e for e in TOT_CARDS if e not in CARDS_FOR_TEST]
CARDS_FOR_VAL = ['jigglypuff+jungle', 'incineroar+hyper'] 
CARDS_FOR_TRAIN = [e for e in TOT_CARDS_NO_TEST if e not in CARDS_FOR_VAL]


def create_ds(data_len = 10, ds_type = 'train'):
    if ds_type == 'train':
        dir_type = CARDS_FOR_TRAIN
    elif ds_type == 'val':
        dir_type = CARDS_FOR_VAL
    elif ds_type == 'test':
        dir_type = CARDS_FOR_TEST
    input_list = []
    label_list = []
    while len(input_list) < data_len:
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
        input_img = np.concatenate((image1,image2), axis = 1)
        input_list.append(input_img)
        label_list.append(label)
    data = [np.array(input_list), np.array(label_list)]
    return data


def define_model():
    cards_input = keras.Input(shape = (7, 10, 1280), name = "reference_img")
    # x = layers.MaxPooling2D(3)(cards_input)
    # x = layers.Conv2D(2048, 3, activation="relu")(cards_input)
    # x = layers.Conv2D(1024, 3, activation="relu")(x)
    # x = layers.Conv2D(1024, 3, activation="relu")(x)
    # x = layers.Conv2D(1024, 3, activation="relu")(x)
    # x = layers.Conv2D(256, 3, activation="relu")(cards_input)
    # # cnn1_output = layers.GlobalMaxPooling2D()(x)
    # cnn1_output = layers.Flatten()(x)
    # card2_input = keras.Input(shape = (7, 5, 1280), name = "new_img")
    # x = layers.MaxPooling2D(3)(x)
    # x = layers.Conv2D(128, 3, activation="relu")(card2_input)
    # x = layers.Conv2D(128, 3, activation="relu")(x)
    # x = layers.GlobalMaxPooling2D()(cards_input)
    x = layers.Flatten()(cards_input)
    x = layers.Dense(64, activation = 'relu')(x)
    x = layers.Dense(64, activation = 'relu')(x)
    x = layers.Dense(64, activation = 'relu')(x)
    x = layers.Dense(32, activation = 'relu')(x)
    x = layers.Dense(32, activation = 'relu')(x)
    x = layers.Dense(16, activation = 'relu')(x)
    match_probability = layers.Dense(1, name="match", activation = 'sigmoid')(x)
    # match_probability = layers.Softmax(x)
    model = keras.Model(inputs = [cards_input], outputs = [match_probability])
    model.compile(optimizer=keras.optimizers.RMSprop(1e-6),
                  loss={'match': keras.losses.BinaryCrossentropy(from_logits=False)},
                  loss_weights=[1.0])
    return model


def cache_data(ds_type = 'train', num_cycles = 500):
    for i in range(num_cycles):
        print(f'Processing {i}-th cycle of {ds_type}')
        if ds_type == 'train':
            ds = create_ds()
        elif ds_type == 'val':
            ds = create_ds(ds_type = 'val')
        elif ds_type == 'test':
            ds = create_ds(ds_type = 'test')
        input_imgs = preprocess_input(ds[0])
        input_ds = base_model(input_imgs)
        with open(os.path.join(CACHE_PATH, f'processed_{ds_type}_{i}.pkl'), 'wb') as f:
            pickle.dump([input_ds, ds[1]], f)


def load_data(ds_type = 'train', num_cycles = 200):
    for i in range(num_cycles):
        print(f'Loading {i}-th cycle of {ds_type}')
        with open(os.path.join(CACHE_PATH, f'processed_{ds_type}_{i}.pkl'), 'rb') as f:
            input_ds, target = pickle.load(f)
        if i == 0:
            total_input_ds = input_ds
            total_target = target
        else:
            total_input_ds = np.append(total_input_ds, input_ds, axis=0)
            total_target = np.append(total_target, target)
    return total_input_ds, total_target
    

def train_model():
    model = define_model()
    keras.utils.plot_model(model, 'pokemon_comparison_cnn.png', show_shapes = True)
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience = 2,
         restore_best_weights=True)
    input_ds, target = load_data()
    input_val, target_val = load_data(ds_type = 'val', num_cycles = 30)
    history = model.fit(
        {'reference_img': input_ds},
        {'match': target},
        epochs = EPOCHS,
        batch_size = 32,
        callbacks = [callback],
        validation_data=({'reference_img': input_val},
                         {'match': target_val})
    )
    # serialize model to JSON
    model_json = model.to_json()
    with open(os.path.join(MODEL_PATH, "model.json"), "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(os.path.join(MODEL_PATH, "model.h5"))
    print("Saved model to disk")
    return model


def main():
    if is_create_ds:
        cache_data()
        cache_data(ds_type = 'val', num_cycles = 30)
        cache_data(ds_type = 'test', num_cycles = 30)
    model = train_model()
    # else:
    #     json_file = open(os.path.join(MODEL_PATH, 'model.json'), 'r')
    #     loaded_model_json = json_file.read()
    #     json_file.close()
    #     model = keras.models.model_from_json(loaded_model_json)
    #     # load weights into new model
    #     model.load_weights(os.path.join(MODEL_PATH, "model.h5"))
    #     print("Loaded model from disk")
    # ds_test = create_ds(ds_type = 'test', data_len = 100)
    input_test, target_test = load_data(ds_type = 'test', num_cycles = 30)
    # input_test_2 = base_model(preprocess_input(ds_test[1]))
    predictions = model.predict({'reference_img': input_test}).flatten()
    preds = np.where(predictions<0.5, 0, 1)
    print(metrics.f1_score(target_test, preds))
    breakpoint()
    # for i in range(len(target_test)):
    #     keras.preprocessing.image.array_to_img(input_test[i]).show()
    #     print(preds[i], input_test[i])
    #     breakpoint()


if __name__=='__main__':
    main()
    # create_ds()
