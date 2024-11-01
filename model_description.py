""" miRNA prediction of a sequence input or a file input containing many sequences
"""
#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import csv
import sys, getopt
import numpy as np
import os
from keras.models import load_model
import tensorflow.keras.backend as K

modelfile = 'models/CNN_model.h5'
modelfile = 'models/RNN_model.h5'
modelfile = 'models/CNNRNN_model.h5'

# check file exist
if not os.path.exists(modelfile):
    print("file doesn't exist!")
    sys.exit(1)

print (f"modelfile={modelfile}")

model = load_model(modelfile)

trainable_count = np.sum([K.count_params(w) for w in model.trainable_weights])
non_trainable_count = np.sum([K.count_params(w) for w in model.non_trainable_weights])

print('Total params: {:,}'.format(trainable_count + non_trainable_count))
print('Trainable params: {:,}'.format(trainable_count))
print('Non-trainable params: {:,}'.format(non_trainable_count))
