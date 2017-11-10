from sklearn.model_selection import train_test_split
import numpy as np

def split():

    fx = np.loadtxt(fname = "small_loan_x.csv", delimiter = ',')
    fy = np.loadtxt(fname = "small_loan_y.csv", delimiter = ',')

    #fx = np.loadtxt(fname = "loan_x_final copy.csv", delimiter = ',')
    #fy = np.loadtxt(fname = "loan_y_final copy.csv", delimiter = ',')

    X = fx[:, 1:]
    y = fy[:, 1:]

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    return X_train, X_test, y_train, y_test