from util import *
from model import *
import pdb
import pprint
from tqdm import tqdm
import numpy as np
import IPython
import soundfile as sf
import matplotlib.pyplot as plt
import librosa
import os
import sys

from tensorflow.keras.models import Sequential
from kapre.time_frequency import Melspectrogram, Spectrogram
from kapre.utils import Normalization2D

from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

from datetime import datetime

data_folder = "data/CLIPPED_VAST/"
label_folder = "data/CLIPPED_VAST/"

namePrefix = sys.argv[1] if len(sys.argv) > 1 else 'four_vtb_classifier_model'
namePrefix += datetime.now().strftime("[%Y-%m-%d-%H-%M-%S]")

data = []
label = []
for i in tqdm(range(1)):
  data.append(load_data_array_from_npy(data_folder + f"dataset_{i}.npy"))
  label.extend(load_data_array_from_npy(label_folder + f"labelset_{i}.npy"))
data = np.vstack(data)

sample_rate = 8000

melspecModel = Sequential()

melspecModel.add(Melspectrogram(n_dft=1024, n_hop=128, input_shape=(1, data.shape[-1]),
                         padding='same', sr=sample_rate, n_mels=80,
                         fmin=40.0, fmax=sample_rate/2, power_melgram=1.0,
                         return_decibel_melgram=True, trainable_fb=False,
                         trainable_kernel=False,
                         name='mel_stft') )

melspecModel.add(Normalization2D(int_axis=0))

# For test
melspecModel.summary()

class_weights = class_weight.compute_class_weight('balanced',
                                                 np.unique(label),
                                                 label)
# convert to dict
assert len(class_weights.shape) == 1
cw_dict = dict()
for cwi in range(len(class_weights)):
    cw_dict[cwi] = class_weights[cwi]

label = to_categorical(label) # Turn to one-hot
data, X_test, label, y_test = train_test_split(data, label, test_size=0.2, random_state=42)
print(f"X_train shape: {data.shape}")
print(f"y_test[0]: {y_test[0]}")

# Create model

model = AttRNNSpeechModel(4, dropout_rate=0.1)
model.compile(optimizer='adam', loss=['categorical_crossentropy'], metrics=['categorical_accuracy'])
# model.compile(optimizer='adam', loss=[multi_category_focal_loss2_fixed], metrics=['categorical_accuracy'])
model.summary()
lrate = LearningRateScheduler(step_decay)

# gc.collect - not necessary here ?

results = model.fit(x = data, y = label,
				epochs=30,
				batch_size=2048,
				shuffle=True,
				validation_data=(X_test, y_test),
				callbacks=[lrate],
				class_weight=cw_dict
				)

os.makedirs(namePrefix)
model.save_weights('results/%s/model.h5' % namePrefix)

print("Training finished")
