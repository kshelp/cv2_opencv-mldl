#1: mnist.py : load MNIST
import gzip
import numpy as np

#1-1
IMAGE_SIZE  =  28
def extract_data(filename, num_images):
    with gzip.open(filename) as bytestream:
        bytestream.read(16)
        buf = bytestream.read(IMAGE_SIZE * IMAGE_SIZE * num_images)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        data = data.reshape(num_images, IMAGE_SIZE, IMAGE_SIZE)
    
    return data

#1-2
def extract_labels(filename, num_images):
  with gzip.open(filename) as bytestream:
    bytestream.read(8)
    buf = bytestream.read(1 * num_images)
    labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int32)
  return labels

#1-3
def ont_hot_encoding(y): # assume that y is 1-D array
    num_arr = np.unique(y)
    n = len(num_arr)  #num_arr.shape[0]
    return np.float32(np.eye(n)[y])    
  
#1-4: Extract gzip files into np arrays.
def load(flatten=False, one_hot=False, normalize = False):
    x_train=extract_data('./data/train-images-idx3-ubyte.gz',  60000)
    y_train=extract_labels('./data/train-labels-idx1-ubyte.gz',60000)
    x_test =extract_data('./data/t10k-images-idx3-ubyte.gz',   10000)
    y_test =extract_labels('./data/t10k-labels-idx1-ubyte.gz', 10000)

    if normalize:
        x_train = x_train/255
        x_test  = x_test/255

    if flatten:
        x_train= x_train.reshape(-1, IMAGE_SIZE*IMAGE_SIZE) # (60000, 784)
        x_test = x_test.reshape(-1, IMAGE_SIZE*IMAGE_SIZE)  # (10000, 784)

    if one_hot:
        y_train = ont_hot_encoding(y_train)  #(60000, 10)
        y_test = ont_hot_encoding(y_test)    #(10000, 10)
    return (x_train, y_train), (x_test, y_test)

 
