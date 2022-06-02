import random
import numpy as np
from numpy.core.fromnumeric import squeeze
from psutil import AIX
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
tf.disable_v2_behavior()
from tensorflow.compat.v1 import keras
from tensorflow.compat.v1.keras import activations
from tensorflow_addons.layers import GroupNormalization
from tensorflow.compat.v1.keras import models
from tensorflow.compat.v1.keras import backend as K
from tensorflow.compat.v1.keras.models import Sequential, Model, load_model
from tensorflow.compat.v1.keras.layers import GlobalAveragePooling2D, concatenate, Add, Multiply, Permute, Softmax, AveragePooling2D, MaxPooling2D, Convolution2D, LeakyReLU, add, Reshape, Lambda, Conv2D, LSTMCell, LSTM, Dense, RepeatVector, TimeDistributed, Input, BatchNormalization, multiply, Concatenate, Flatten, Activation, dot, Dot, Dropout
from tensorflow.compat.v1.keras.utils import to_categorical
from tensorflow.compat.v1.keras import losses



class ExplorationTarget(keras.Model):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv2D(32, 8, (4,4), activation='relu', padding='same', kernel_initializer=keras.initializers.glorot_uniform(seed=2))
        self.conv2 = Conv2D(64, 4, (2,2), activation='relu', padding='same', kernel_initializer=keras.initializers.glorot_uniform(seed=2))
        self.conv3 = Conv2D(16, 3, (1,1), activation='relu', padding='same', kernel_initializer=keras.initializers.glorot_uniform(seed=2))
        self.flatten = Flatten()
        self.dense1 = Dense(512, activation='linear', kernel_initializer=keras.initializers.glorot_uniform(seed=2))

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.dense1(x)
        return x


class ExplorationTrain(keras.Model):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv2D(32, 8, (4,4), activation='relu', padding='same', kernel_initializer=keras.initializers.glorot_uniform(seed=3))
        self.conv2 = Conv2D(64, 4, (2,2), activation='relu', padding='same', kernel_initializer=keras.initializers.glorot_uniform(seed=3))
        self.conv3 = Conv2D(16, 3, (1,1), activation='relu', padding='same', kernel_initializer=keras.initializers.glorot_uniform(seed=3))
        self.flatten = Flatten()
        self.dense1 = Dense(512, activation='linear', kernel_initializer=keras.initializers.glorot_uniform(seed=3))

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.dense1(x)
        return x


class rnd:
    def __init__(self):
        self.target_model = ExplorationTarget()
        self.train_model = ExplorationTrain()
    
    def cal(self, state):
        # if state.shape[0] != 200:
        #     state = state[:200, :, :]
        state = (state - 0) / 255.0
        state = np.expand_dims(state, axis=0)
        target = self.target_model.predict(state)
        pred = self.train_model.predict(state)
        rew = 

