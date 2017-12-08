from sklearn.model_selection import train_test_split
import numpy as np
import os.path
import pandas as pd

FILE_INPUT_X = "small_loan_x.csv"
FILE_INPUT_Y= "small_loan_y.csv"

TRAINING_SPLIT_X="training_split_x.csv"
TESTING_SPLIT_X="testing_split_x.csv"

TRAINING_SPLIT_Y="training_split_y.csv"
TESTING_SPLIT_Y="testing_split_y.csv"

PD = 0
NP = 1
DEFAULT_VOLUME = 10000
ORIGINAL_FILE_LENGTH = 20000
def split(outputType,volume=DEFAULT_VOLUME):

    if volume != DEFAULT_VOLUME or not os.path.exists(TRAINING_SPLIT_X) or not os.path.exists(TESTING_SPLIT_X) or not os.path.exists(TRAINING_SPLIT_Y) or not os.path.exists(TESTING_SPLIT_Y):
        print "Writing splits"
        fx = np.loadtxt(fname=FILE_INPUT_X, delimiter=',')
        fy = np.loadtxt(fname=FILE_INPUT_Y, delimiter=',')
        X = fx[ORIGINAL_FILE_LENGTH-volume:, 1:]
        y = fy[ORIGINAL_FILE_LENGTH-volume:, 1:]
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

        np.savetxt(TRAINING_SPLIT_X,X_train,delimiter=',')
        np.savetxt(TRAINING_SPLIT_Y, y_train, delimiter=',')
        np.savetxt(TESTING_SPLIT_X, X_test, delimiter=',')
        np.savetxt(TESTING_SPLIT_Y, y_test, delimiter=',')
        #

        # fx = pd.read_csv("loan_x_default_last.csv", header=None)
        # fy = pd.read_csv("loan_y_default_last.csv", header=None)

        # fx = pd.read_csv("small_loan_x.csv", header=None)
        # fy = pd.read_csv("small_loan_y.csv", header=None)


    # X = fx.iloc[10000:, 1:]
    # y = fy.iloc[10000:, 1:]


    if outputType == PD:
        X_train = pd.read_csv(TRAINING_SPLIT_X, header=None)
        X_test = pd.read_csv(TESTING_SPLIT_X, header=None)
        y_train = pd.read_csv(TRAINING_SPLIT_Y, header=None)
        y_test = pd.read_csv(TESTING_SPLIT_Y, header=None)
    elif outputType == NP:
        X_train = np.loadtxt(fname=TRAINING_SPLIT_X, delimiter=',')
        X_test = np.loadtxt(fname = TESTING_SPLIT_X,delimiter = ',')
        y_train = np.loadtxt(fname=TRAINING_SPLIT_Y, delimiter=',')
        y_test = np.loadtxt(fname=TESTING_SPLIT_Y, delimiter=',')


    return X_train, X_test, y_train, y_test

