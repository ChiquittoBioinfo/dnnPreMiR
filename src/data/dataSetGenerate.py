"""Read the data from csv file as dataframe""" 
import pandas as pd
from sklearn.utils import shuffle

# read the csv file and merged into the dataset of dataframe
def read_new_csv(positive,negative,doshuffle):
    try:       
        hsa = pd.read_csv(positive)
        pseudo = pd.read_csv(negative) if not negative is None else None
    except IOError:
        print("Exception:hsa_new.csv or pseudo_new.csv file does not exist!")
        exit(2)

    # merge the positive and negative data into a dataset
    if negative is None:
        dataset = pd.concat([hsa])
    else:
        dataset = pd.concat([hsa, pseudo])
    # shuffle the order
    if doshuffle:
        dataset = shuffle(dataset)

    print("dataset is prepared!")
    return dataset

if __name__ == '__main__':
    positive = "hsa_new.csv"
    negative = "pseudo_new.csv"
    dataset = read_new_csv(positive,negative)
