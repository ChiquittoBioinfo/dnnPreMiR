# DataAugmentation directory

```bash
ln -s /home/alissonchiquitto/work/DataAugmentation DataAugmentation
```

# Criar o ambiente CONDA

```bash
conda create python=3.10.9 --name dnnPreMiR -y

conda activate dnnPreMiR

conda install -c conda-forge cudatoolkit=11.2.2 cudnn=8.1.0 -y
# pip install --upgrade keras
python -m pip install viennarna
pip install pandas
pip install scikit-learn
```

# Verify install of GPU

```bash
python3 -c "import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'; import tensorflow as tf; print('Num GPUs Available: ', len(tf.config.list_physical_devices('GPU')))"
```

# To remove the env

```bash
conda remove --name dnnPreMiR --all
```

# Convert CSV para FASTA

```bash
awk 'BEGIN {FS=","} {print ">" $1 "\n" $2}' testData/chiquitto/chiquitto.csv > testData/chiquitto/chiquitto.fa
```

# Training a CNN:

```bash
cd src/CNN
conda activate dnnPreMiR
python CNNTrain_chiquitto.py --pos ../data/hsa_new.csv --neg ../data/pseudo_new.csv --output ../../models
```

# Training a RNN:

```bash
cd src/RNN
conda activate dnnPreMiR
python RNNTrain_chiquitto.py --pos ../data/hsa_new.csv --neg ../data/pseudo_new.csv --output ../../models
```

# Training a CNN_RNN:

```bash
cd human_pre_miRNA/CNN_RNN
conda activate dnnPreMiR
python CNNRNNTrain_chiquitto.py --pos ../data/hsa_new.csv --neg ../data/pseudo_new.csv --output ../../models
```

# Predicting

```bash
python isPreMiR_chiquitto.py -i input.fa -o output.csv -m model.h5
python isPreMiR_chiquitto.py -i testData/chiquitto/chiquitto.fa -o temp/results_cnn.csv -m models/CNN_model.h5
python isPreMiR_chiquitto.py -i testData/chiquitto/chiquitto.fa -o temp/results_rnn.csv -m models/RNN_model.h5
python isPreMiR_chiquitto.py -i testData/chiquitto/chiquitto.fa -o temp/results_cnnrnn.csv -m models/CNNRNN_model.h5
```


from keras import backend as K
K.tensorflow_backend._get_available_gpus()