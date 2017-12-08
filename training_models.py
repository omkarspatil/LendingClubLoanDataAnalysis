from sklearn import svm
import model_names
import numpy as np

models_dict={
    model_names.LINEAR_MODEL:svm.SVC,
    model_names.SVC_MODEL:svm.SVC,
    model_names.RBF_MODEL:svm.SVC,
    model_names.POLYNOMIAL_MODEL:svm.SVC,
    model_names.LINEAR_SVC_MODEL:svm.LinearSVC
}


def training(model, param_map, X, y):
    model_t = models_dict[model](**param_map)
    model_t.fit(X,y)
    print model_t.C
    #print model_t.kernel
    return model_t


def testing_error(model, X_test, y_test):
    error_sum=0
    for i in xrange(len(X_test)):
        #print X_test[i]
        #print y_test[i:i+1].iat[0,0]
        #print model.predict(X_test[i:i+1])[0]
        #print y_test.iloc[i, 0:]
        #error_sum += (y_test[i:i+1].iat[0,0]-model.predict(X_test[i:i+1])[0])** 2
        error_sum += diff_error(y_test[i],model.predict(np.reshape(X_test[i,:], (-1, X_test.shape[1])) )[0])
    #print "Error sum",error_sum
    return error_sum/float(len(X_test))


def diff_error(y,pred):
    return 1 if y!=pred else 0