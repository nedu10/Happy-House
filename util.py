import math
import numpy as np
import h5py
import tensorflow as tf
# from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Conv2DTranspose, concatenate, ZeroPadding2D, Dense
 


def load_happy_dataset():
    train_dataset = h5py.File('datasets/train_happy.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    # print("train_set_x_orig >> ", train_set_x_orig.shape)
    # print("train_set_y_orig >> ", train_set_y_orig.shape)

    test_dataset = h5py.File('datasets/test_happy.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    # print("test_set_x_orig >> ", test_set_x_orig.shape)
    # print("test_set_y_orig >> ", test_set_y_orig.shape)

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    # print("train_set_y_orig >> ", train_set_y_orig.shape)
    # print("test_set_y_orig >> ", test_set_y_orig.shape)
    # print("classes >> ", classes.shape)

    # train_set_x_orig >>  (600, 64, 64, 3)
    # train_set_y_orig >>  (600,)
    # test_set_x_orig >>  (150, 64, 64, 3)
    # test_set_y_orig >>  (150,)
    # train_set_y_orig >>  (1, 600)
    # test_set_y_orig >>  (1, 150)
    # classes >>  (2,)
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes
