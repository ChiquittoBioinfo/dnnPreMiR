""" Train the CNN model using dataset
"""

# python CNNRNNTrain_chiquitto.py --pos ../data/hsa_new.csv --neg ../data/pseudo_new.csv --output models

import sys
sys.path.append("../data") 
from CNNModel import CNN_model
import dataSetPartition
import keras
import os

import csv
from CNNTrain_args import process_argv

opts = process_argv()
print(f"opts={opts}")

def CNN_train(x_dataset,y_dataset):
    model = CNN_model()

    # if os.path.exists("CNN_model_preTrained.h5"):
    #     print("load the weights")
    #     model.load_weights("CNN_model_preTrained.h5")

    model.fit(x_dataset,y_dataset,batch_size = 200, epochs = 600,\
          validation_split = 0.2)
    print("model train over")
    return model

if __name__ == "__main__":
    positive = opts['pos']
    negative = opts['neg']

    x_train_dataset,y_train_dataset,x_test_dataset,y_test_dataset = \
      dataSetPartition.train_test_partition(positive,negative,doshuffle=False)
      
    model = CNN_train(x_train_dataset,y_train_dataset)

    modelfile = f"{opts['output']}/CNN_model.h5"
    model.save(modelfile)

    print(f"The model is saved as {modelfile}")

