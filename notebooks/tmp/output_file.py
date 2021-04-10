
import os
import builtins
import sys
import numpy as np
import pandas as pd
import io


import plaidml.keras

plaidml.keras.install_backend()
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
# os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=opencl0:2,floatX=float32"
# os.environ["KERAS_BACKEND"] = "theano"
# os.environ["KERAS_BACKEND"] = "tensorflow"

import tensorflow as tf
import keras

from google.cloud import storage
from urllib.parse import urlparse


from PIL import Image
from tqdm import tqdm
from matplotlib import pyplot as plt


from keras.utils import Sequence

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model
from keras.layers import (
    Activation,
    Dropout,
    Flatten,
    Dense,
    Input,
    Conv2D,
    ConvLSTM2D,
    MaxPooling2D,
    BatchNormalization,
)
from keras import metrics
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras import backend as K


from sklearn.metrics import f1_score

from keras.utils import multi_gpu_model
from tensorflow.python.client import device_lib


from multiprocess import Pool
DIR = r"../data/raw/human-protein-atlas-image-classification"
# DIR = f"gs://{BUCKET}/data/human-protein-atlas-image-classification"
# BUCKET=os.getenv("BUCKET")
batch_size = 16
img_height = 200
img_width = 200
BATCH_SIZE = 16

SEED = 777
SHAPE = (512, 512, 4)

VAL_RATIO = 0.1  # 10 % as validation
DEBUG = True


THRESHOLD = np.float64(
    0.05
)  # due to different cost of True Positive vs False Positive, this is the probability threshold to predict the class as 'yes'


def getTrainDataset():

    path_to_train = DIR + "/train/"
    data = pd.read_csv(DIR + "/train.csv")

    paths = []
    labels = []

    for name, lbl in zip(data["Id"], data["Target"].str.split(" ")):
        y = np.zeros(28)
        for key in lbl:
            y[int(key)] = 1
        paths.append(os.path.join(path_to_train, name))
        labels.append(y)

    return np.array(paths), np.array(labels)


def getTestDataset():

    path_to_test = DIR + "/test/"
    data = pd.read_csv(DIR + "/sample_submission.csv")

    paths = []
    labels = []

    for name in data["Id"]:
        y = np.ones(28)
        paths.append(os.path.join(path_to_test, name))
        labels.append(y)

    return np.array(paths), np.array(labels)


# credits: https://github.com/keras-team/keras/blob/master/keras/utils/data_utils.py#L302
# credits: https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly


class ProteinDataGenerator(keras.utils.Sequence):
    def __init__(
        self, paths, labels, batch_size, shape, shuffle=False, use_cache=False
    ):
        self.paths, self.labels = paths, labels
        self.batch_size = batch_size
        self.shape = shape
        self.shuffle = shuffle
        self.use_cache = use_cache
        if use_cache == True:
            self.cache = np.zeros((paths.shape[0], shape[0], shape[1], shape[2]))
            self.is_cached = np.zeros((paths.shape[0]))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.paths) / float(self.batch_size)))

    def len(self):
        return int(np.ceil(len(self.paths) / float(self.batch_size)))

    def __getitem__(self, idx):
        indexes = self.indexes[idx * self.batch_size : (idx + 1) * self.batch_size]

        paths = self.paths[indexes]
        X = np.zeros((paths.shape[0], self.shape[0], self.shape[1], self.shape[2]))
        # Generate data
        if self.use_cache == True:
            X = self.cache[indexes]
            for i, path in enumerate(paths[np.where(self.is_cached[indexes] == 0)]):
                image = self.__load_image(path)
                self.is_cached[indexes[i]] = 1
                self.cache[indexes[i]] = image
                X[i] = image
        else:
            for i, path in enumerate(paths):
                X[i] = self.__load_image(path)

        y = self.labels[indexes]

        return X, y

    def on_epoch_end(self):

        # Updates indexes after each epoch
        self.indexes = np.arange(len(self.paths))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __iter__(self):
        """Create a generator that iterate over the Sequence."""
        for item in (self[i] for i in range(len(self))):
            yield item

    def __load_image(self, path):
        try:
            # print(path)
            R = Image.open(path + "_red.png")
            G = Image.open(path + "_green.png")
            B = Image.open(path + "_blue.png")
            Y = Image.open(path + "_yellow.png")
        except:
            o = urlparse(path)
            storage_client = storage.Client("home-225723")
            bucket = storage_client.get_bucket(o.netloc)
            # print(path)
            R = Image.open(
                io.BytesIO(bucket.blob(o.path[1:] + "_red.png").download_as_string())
            )
            G = Image.open(
                io.BytesIO(bucket.blob(o.path[1:] + "_green.png").download_as_string())
            )
            B = Image.open(
                io.BytesIO(bucket.blob(o.path[1:] + "_blue.png").download_as_string())
            )
            Y = Image.open(
                io.BytesIO(bucket.blob(o.path[1:] + "_yellow.png").download_as_string())
            )

        im = np.stack((np.array(R), np.array(G), np.array(B), np.array(Y)), -1)

        im = np.divide(im, 255)

        return im

# credits: https://www.kaggle.com/guglielmocamporese/macro-f1-score-keras
# https://stackoverflow.com/questions/58931078/how-to-replace-certain-parts-of-a-tensor-on-the-condition-in-keras/58931377#58931377


def f1(y_true, y_pred, dtype="float32"):
    dtype = "float32"
    y_pred = K.cast(y_pred, dtype)
    y_true = K.cast(y_true, dtype)
    y_pred = K.cast(K.greater(K.clip(y_pred, 0, 1), THRESHOLD), dtype)
    tp = K.sum(y_true * y_pred, axis=0)
    tn = K.sum(K.cast((1 - y_true) * (1 - y_pred), dtype), axis=0)
    fp = K.sum(K.cast((1 - y_true) * y_pred, dtype), axis=0)
    fn = K.sum(K.cast(y_true * (1 - y_pred), dtype), axis=0)

    p = K.cast(tp / (tp + fp + K.epsilon()), dtype)
    r = K.cast(tp / (tp + fn + K.epsilon()), dtype)

    diff = 2 * p * r
    suma = p + r + K.epsilon()
    d0 = K.equal(diff, 0)
    s0 = K.equal(suma, 0)
    # sum zeros are replaced by ones on division
    rel_dev = diff / K.switch(s0, K.ones_like(suma), suma)
    rel_dev = K.switch(d0 & s0, K.zeros_like(rel_dev), rel_dev)
    try:
        # ~ is the bitwise complement operator in python which essentially calculates (-x - 1)
        rel_dev = K.switch(-d0 - 1 & s0, K.sign(diff), rel_dev)
    except:
        rel_dev = K.switch(~d0 & s0, K.sign(diff), rel_dev)

    f1 = rel_dev

    return K.cast(K.mean(f1), dtype)


# some basic useless model
def create_model(input_shape):
    model = Sequential()
    model.add(Conv2D(8, (3, 3), activation="relu", input_shape=input_shape))
    model.add(BatchNormalization(axis=-1))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(16, (3, 3), activation="relu"))
    model.add(BatchNormalization(axis=-1))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(32, (3, 3), activation="relu"))
    model.add(BatchNormalization(axis=-1))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(BatchNormalization(axis=-1))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(128, (3, 3), activation="relu"))
    model.add(BatchNormalization(axis=-1))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(256, (3, 3), activation="relu"))
    model.add(BatchNormalization(axis=-1))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dropout(0.5))
    # model.add(Dense(28))
    # model.add(Activation('relu'))
    # model.add(Dropout(0.1))
    model.add(Dense(28))
    model.add(Activation("sigmoid"))

    return model

model = create_model(SHAPE)

# model.compile(loss='mse', optimizer='sgd')
model.compile(loss="binary_crossentropy", optimizer=Adam(0.0001), metrics=["acc", f1])

DEBUG = False

paths, labels = getTrainDataset()

# divide to
keys = np.arange(paths.shape[0], dtype=np.int)
np.random.seed(SEED)
np.random.shuffle(keys)
lastTrainIndex = int((1 - VAL_RATIO) * paths.shape[0])

if DEBUG == True:  # use only small subset for debugging, Kaggle's RAM is limited
    pathsTrain = paths[0:256]
    labelsTrain = labels[0:256]
    pathsVal = paths[lastTrainIndex : lastTrainIndex + 256]
    labelsVal = labels[lastTrainIndex : lastTrainIndex + 256]
    use_cache = True
else:
    pathsTrain = paths[0:lastTrainIndex]
    labelsTrain = labels[0:lastTrainIndex]
    pathsVal = paths[lastTrainIndex:]
    labelsVal = labels[lastTrainIndex:]
    use_cache = True

# print(paths.shape, labels.shape)
# print(pathsTrain.shape, labelsTrain.shape, pathsVal.shape, labelsVal.shape)

file_name = os.getcwd() + "/base.model"
# test_dataset = h5py.File(file_name, "r")


tg = ProteinDataGenerator(
    pathsTrain, labelsTrain, BATCH_SIZE, SHAPE, use_cache=use_cache
)
vg = ProteinDataGenerator(pathsVal, labelsVal, BATCH_SIZE, SHAPE, use_cache=use_cache)


# https://keras.io/callbacks/#modelcheckpoint
checkpoint = ModelCheckpoint(
    file_name,
    monitor="val_f1",
    verbose=1,
    save_best_only=True,
    save_weights_only=False,
    mode="max",
    period=1,
)

epochs = 3

if DEBUG == True:
    use_multiprocessing = True  # DO NOT COMBINE WITH CACHE!
    workers = 1  # DO NOT COMBINE WITH CACHE!
else:
    use_multiprocessing = True
    workers = 1
# import output_file
# from multiprocess import Pool

if __name__ == "__main__":
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
    #with Pool(workers) as p:
    hist = model.fit_generator(
            tg,
            steps_per_epoch=len(tg),
            validation_data=vg,
            validation_steps=8,
            epochs=epochs,
            use_multiprocessing=use_multiprocessing,  # you have to train the model on GPU in order to this to be benefitial
            workers=workers,  # you have to train the model on GPU in order to this to be benefitial
            verbose=1,
            callbacks=[checkpoint],
        )

    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    ax[0].set_title("loss")
    ax[0].plot(hist.epoch, hist.history["loss"], label="Train loss")
    ax[0].plot(hist.epoch, hist.history["val_loss"], label="Validation loss")
    ax[1].set_title("acc")
    ax[1].plot(hist.epoch, hist.history["f1"], label="Train F1")
    ax[1].plot(hist.epoch, hist.history["val_f1"], label="Validation F1")
    ax[0].legend()
    ax[1].legend()
