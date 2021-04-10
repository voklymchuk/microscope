
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
