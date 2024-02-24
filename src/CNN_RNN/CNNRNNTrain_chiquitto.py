""" Train the BiRNN model using dataset
"""

# python CNNRNNTrain_chiquitto.py --pos ../data/hsa_new.csv --neg ../data/pseudo_new.csv --output models

import sys
sys.path.append("../data") 
from CNNRNNModel import CNN_RNN_model
import dataSetPartition
import keras
import os

import csv
from CNNRNNTrain_args import process_argv

opts = process_argv()
print(f"opts={opts}")

def CNNRNN_train(x_dataset,y_dataset):
    model = CNN_RNN_model()
    
    # # transfer learning
    # if os.path.exists("CNNRNN_model_preTrained.h5"):
    #     print("load the weights")
    #     model.load_weights("CNNRNN_model_preTrained.h5")

    model.fit(x_dataset,y_dataset,batch_size = 200, epochs = 300,\
          validation_split = 0.2)
    print("model train over")
    return model

if __name__ == "__main__":
    positive = opts['pos']
    negative = opts['neg']

    x_train_dataset,y_train_dataset,x_test_dataset,y_test_dataset = \
      dataSetPartition.train_test_partition(positive,negative,doshuffle=False)

    model = CNNRNN_train(x_train_dataset,y_train_dataset)

    modelfile = f"{opts['output']}/CNNRNN_model.h5"
    model.save(modelfile)

    print(f"The model is saved as {modelfile}")

