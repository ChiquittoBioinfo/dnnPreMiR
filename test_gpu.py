# import tensorflow as tf

# print(f'\nTensorflow version = {tf.__version__}\n')
# print(f'\n{tf.config.list_physical_devices("GPU")}\n')

from keras import backend as K
K.tensorflow_backend._get_available_gpus()