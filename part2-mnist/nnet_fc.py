#! /usr/bin/env python

import _pickle as cPickle, gzip
import numpy as np
from tqdm import tqdm
import torch
import torch.autograd as autograd
import torch.nn.functional as F
import torch.nn as nn
import sys
sys.path.append("..")
import utils
from utils import *
from train_utils import batchify_data, run_epoch, train_model

def main():
    # Load the dataset
    num_classes = 10
    X_train, y_train, X_test, y_test = get_MNIST_data()

    # Split into train and dev
    dev_split_index = int(9 * len(X_train) / 10)
    X_dev = X_train[dev_split_index:]
    y_dev = y_train[dev_split_index:]
    X_train = X_train[:dev_split_index]
    y_train = y_train[:dev_split_index]

    permutation = np.array([i for i in range(len(X_train))])
    np.random.shuffle(permutation)
    X_train = [X_train[i] for i in permutation]
    y_train = [y_train[i] for i in permutation]

    # Split dataset into batches
    batch_size = 32
    train_batches = batchify_data(X_train, y_train, batch_size)
    dev_batches = batchify_data(X_dev, y_dev, batch_size)
    test_batches = batchify_data(X_test, y_test, batch_size)

    #################################
    ## Model specification TODO
    model = nn.Sequential(
              nn.Linear(784, 128),
              nn.LeakyReLU(negative_slope = 0.01),
              nn.Linear(128, 10),
            )
    lr=0.1
    momentum=0
    ##################################

    train_model(train_batches, dev_batches, model, lr=lr, momentum=momentum)

    ## Evaluate the model on test data
    loss, accuracy = run_epoch(test_batches, model.eval(), None)

    print ("Loss on test set:"  + str(loss) + " Accuracy on test set: " + str(accuracy))


if __name__ == '__main__':
    # Specify seed for deterministic behavior, then shuffle. Do not change seed for official submissions to edx
    np.random.seed(12321)  # for reproducibility
    torch.manual_seed(12321)  # for reproducibility
    main()
    ## baseline :  0.9204727564102564    0.932487
    ## batch size 64 : 0.9314903846153846    0.940020
    ## lr = 0.01 : 0.9206730769230769    0.934492
    ## momentum = 0.9 : 0.8828125        0.895722
    ## LeakyRelu : 0.9207732371794872   0.931985

    ## hidden 10 -> 128 units
    ## baseline : 0.9767628205128205    0.977440
    ## batch size 64 : 0.9745592948717948    0.976983
    ## lr = 0.01 : 0.9427083333333334    0.955047
    ## momentum = 0.9 : 0.9596354166666666    0.963402
    ## LeakyRelu : 0.9771634615384616    0.978777