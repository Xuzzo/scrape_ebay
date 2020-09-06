import os, sys
import pickle
import random
from tensorflow import keras
import tensorflow as tf
import numpy as np
from PIL import Image



class TrainUtils:
    def __init__(self, data_path, model_path):
        self.data_path = data_path
        self.tot_cards = os.listdir(data_path)
        self.tot_cards.remove('.DS_Store')
        if os.path.exists(os.path.join(model_path, 'tr_val_test.pkl')) and False:
            with open(os.path.join(model_path, 'tr_val_test.pkl'), 'rb') as f:
                self.cards_for_train, self.cards_for_val, self.cards_for_test, self.tot_cards_not_test = pickle.load(f)
        else:
            self.cards_for_test = random.sample(self.tot_cards, 2)
            self.tot_cards_not_test = [e for e in self.tot_cards if e not in self.cards_for_test]
            self.cards_for_val = random.sample(self.tot_cards_not_test, 2)
            self.cards_for_train = [e for e in self.tot_cards_not_test if e not in self.cards_for_val]
            with open(os.path.join(model_path, 'tr_val_test.pkl'), 'wb') as f:
                    pickle.dump([self.cards_for_train, self.cards_for_val, self.cards_for_test, self.tot_cards_not_test], f)


    def path_to_ds(self, ds_type):
        if ds_type == 'train':
            dir_type = self.cards_for_train
        elif ds_type == 'val':
            dir_type = self.cards_for_val
        elif ds_type == 'test':
            dir_type = self.cards_for_test
        return dir_type

    def return_matched(self, dir_type):
        while True:
            card1_type = random.choice(dir_type)
            card1_dir = os.path.join(self.data_path, card1_type, 'true')
            card2_dir = card1_dir
            card1, card2 = random.sample(os.listdir(card1_dir), 2)
            if card1 == '.DS_Store':continue
            if card2 == '.DS_Store':continue
            break
        card1_path = os.path.join(card1_dir, card1)
        card2_path = os.path.join(card2_dir, card2)
        return card1_path, card2_path

    def return_unmatched(self, dir_type):
        while True:
            card1_type = random.choice(dir_type)
            card1_dir = os.path.join(self.data_path, card1_type, 'true')
            is_internal = bool(random.getrandbits(1))
            card2_dir = os.path.join(self.data_path, card1_type, 'false')
            if (not is_internal) or len(os.listdir(card2_dir))==0:
                while True:
                    card2_type = random.choice(self.tot_cards_not_test)
                    if card2_type!= card1_type: break
                card2_dir = os.path.join(self.data_path, card2_type, 'true')
            card1 = random.choice(os.listdir(card1_dir))
            card2 = random.choice(os.listdir(card2_dir))
            if card1 == '.DS_Store':continue
            if card2 == '.DS_Store':continue
            break
        card1_path = os.path.join(card1_dir, card1)
        card2_path = os.path.join(card2_dir, card2)
        return card1_path, card2_path


    def create_ds(self, data_len=100, ds_type = 'train', input_shape = (225, 225)):
        dir_type = self.path_to_ds(ds_type)
        reference_img = []
        new_img = []
        label_list = []
        counter_lim = 1_000_000
        counter = 0
        while (len(reference_img) < data_len) and (counter < counter_lim):
            counter +=1
            is_match = bool(random.getrandbits(1))
            if is_match:
                card1_path, card2_path = self.return_matched(dir_type)
                label = 1
            else:
                card1_path, card2_path = self.return_unmatched(dir_type)
                label = 0
            image1 = Image.open(card1_path)
            image1 = image1.resize((225, 225), Image.ANTIALIAS)
            image1 = np.array(image1)
            image2 = Image.open(card2_path)
            image2 = image2.resize((225, 225), Image.ANTIALIAS)
            image2 = np.array(image2)
            reference_img.append(image1)
            new_img.append(image2)
            label_list.append(label)
        data = [np.array(reference_img), np.array(new_img), np.array(label_list)]
        return data
