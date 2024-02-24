""" miRNA prediction of a sequence input or a file input containing many sequences
"""
#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import csv
import sys, getopt
import numpy as np
import os
from keras.models import load_model
#import sys
#sys.path.append("./src/data")
#from dataVectorization import transform_xdata
#from pandas import Series


x_cast = {"A.":[1,0,0,0,0,0,0,0,0,0,0,0],"U.":[0,1,0,0,0,0,0,0,0,0,0,0],\
          "G.":[0,0,1,0,0,0,0,0,0,0,0,0],"C.":[0,0,0,1,0,0,0,0,0,0,0,0],\
          "A(":[0,0,0,0,1,0,0,0,0,0,0,0],"U(":[0,0,0,0,0,1,0,0,0,0,0,0],\
          "G(":[0,0,0,0,0,0,1,0,0,0,0,0],"C(":[0,0,0,0,0,0,0,1,0,0,0,0],\
          "A)":[0,0,0,0,0,0,0,0,1,0,0,0],"U)":[0,0,0,0,0,0,0,0,0,1,0,0],\
          "G)":[0,0,0,0,0,0,0,0,0,0,1,0],"C)":[0,0,0,0,0,0,0,0,0,0,0,1] }


def usage():
    print("""
          USAGE:
          python isPreMiR.py -s RNAsequence 
          for example: python isPreMiR.py -s CUCCGGUGCCUACUGAGCUGAUAUCAGUUCUCAUUUUACACACUGGCUCAGUUCAGCAGGAACAGGA 

          python isPreMiR_chiquitto.py -i inputFilePath -o outputFilePath -m modelFilePath

          """)
def seq_process(seq):
    # remove line break   
    seq = seq.strip('\n')
    # remove sapce
    seq =seq.replace(' ', '')
    # onvert the string to all uppercase
    seq = seq.upper()
    # print(seq)
    # check correctness of the RNA sequence
    for char in seq:
        #print (char)
        if char not in ["A","U","G","C"]:
            print("Please input the right RNA sequence")
            exit(1)
    return seq


def parse_opt(argv):
    infile = ''
    outfile = ''
    seq = ''
    try:
        opts, args = getopt.getopt(argv,"hs:i:o:m:",["sequence=","infile=","outfile=","model="])
    except getopt.GetoptError:
        usage()
        sys.exit(2)
    if len(opts) < 1:
        usage()
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print (usage())
            sys.exit(0)
        elif opt in ("-s", "--sequence"):
            seq = arg
            seq = seq_process(seq)
        elif opt in ("-i","--infile"):
            infile = arg
        elif opt in ("-o","--outfile"):
            outfile = arg
        elif opt in ("-m","--model"):
            model = arg
    return seq,infile,outfile,model

def transform_seq_struct(seq_struct):
    SEQ_LEN = 180
    seq_struct_vector = []
    if len(seq_struct) > SEQ_LEN:
        seq_struct = seq_struct[:SEQ_LEN]
    for item in seq_struct:
        seq_struct_vector.append(x_cast[item])
    # padding to SEQ_LEN
    m_len = len(seq_struct_vector)
    for i in range(SEQ_LEN-m_len):
        seq_struct_vector.append([0,0,0,0,0,0,0,0,0,0,0,0])
    return seq_struct_vector
     

def predict_results(seq_struct_vector, modelfile):
    # reload the model
    model = load_model(modelfile)
    # prediction
    result =  model.predict(seq_struct_vector)

    return result


def main(argv):
    #parse_opt(argv)
    seq,infile,outfile,modelfile = parse_opt(argv)
    
    # check file exist
    if not os.path.exists(infile):
        print("file doesn't exist!")
        sys.exit(1)

    print (f"infile={infile}")
    print (f"outfile={outfile}")
    print (f"modelfile={modelfile}")

    # calculate the second structure
    os.system("./bin/RNAfold -i " + infile + " --noPS" \
                + "> ./temp/temp_infile_seq_struct")

    name_list = []
    seq_list = []
    seq_struct_list = []
    seq_struct_vector_list = []

    # open and read the generated file containing the second structure into lists
    print("read the infile start:")
    fd = open("./temp/temp_infile_seq_struct","r")
    while True:
        name = fd.readline()
        if name:
            name_list.append(name.strip().strip(">"))

            seq_struct = []
            seq = fd.readline().strip().upper().replace('T', 'U')
            seq_list.append(seq)
            struct = fd.readline().strip()
            for index in range(len(seq)):
                seq_struct.append(seq[index]+struct[index])
            seq_struct_list.append(seq_struct)
        else:
            break
    fd.close()    
    
    # transform the seq_struct into vector
    for seq_struct in seq_struct_list:
        seq_struct_vector = transform_seq_struct(seq_struct)
        #print(seq_struct_vector)
        seq_struct_vector_list.append(seq_struct_vector)
        
    # transform to numpy array
    seq_struct_vector_array = np.array(seq_struct_vector_list)
    # make sure the dimension of input data 
    if len(seq_struct_vector_array) == 1: 
        seq_struct_vector_array = seq_struct_vector_array.reshape(1,180,12)
    # prediction results 
    print("prediction start:")
    result = predict_results(seq_struct_vector_array, modelfile=modelfile)
    #print("result:{}".format(result))
    # print(type(result))
    # write to output file or print out
    prediction = np.argmax(result,axis = 1) 
    #print("prediction:{}".format(prediction))
    if outfile:
        fieldnames = ['id', 'seq', 'class']

        csvfile_writer = open(outfile, 'w', newline='')
        csvwriter = csv.DictWriter(csvfile_writer, fieldnames=fieldnames)
        csvwriter.writeheader()

        for i in range(len(prediction)):
            class_val = prediction[i] # zero|true is positive class
            csvwriter.writerow({ 'id': name_list[i], 'seq': seq_list[i], 'class': class_val })

        csvfile_writer.close()
    else:
        for i in range(len(name_list)):
            print(name_list[i])
            print(seq_list[i])
            if prediction[i] == 0:
                print("True")
            else:
                print("False")
            print("===========================")

    print("prediction finished!")
    exit(0)
       

if __name__ == "__main__":
   main(sys.argv[1:])
