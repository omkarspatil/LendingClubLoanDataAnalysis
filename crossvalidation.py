import pandas as pd
import model_names
import training_models
import sys
import numpy as np
import split
import matplotlib.pyplot as plt
import recursive_feature_selection
import random
import os


class train_model:
    def __init__(self,name,fixed_params,hyper_params):
        self.name=name
        self.fixed_params=fixed_params
        self.hyper_params=hyper_params

    def get_hyper_param(self,param):
        return self.hyper_params[param]

train_model_objects={
    model_names.LINEAR_MODEL: train_model(model_names.LINEAR_MODEL,{"kernel": "linear"},{"C" : [x/1000000.0 for x in range(40, 200, 30)]}),
    model_names.LINEAR_SVC_MODEL: train_model(model_names.LINEAR_SVC_MODEL, {}, {"C": [x / 100.0 for x in range(1, 100, 5)]}),
    model_names.RBF_MODEL:train_model(model_names.RBF_MODEL,{"kernel":"rbf"},{"C" : [1000000000000, 0.000000000000004,0.2,0.4,0.6,1]}),
    #model_names.POLYNOMIAL_MODEL:train_model(model_names.POLYNOMIAL_MODEL,{"kernel":"poly","degree":3},{"C" : [0.000000000000001,0.0005]})
}


def kfold_cross_validation(k):

    feature_map = recursive_feature_selection.select_k_features(7)
    #print feature_map.keys()

    df,x_test,y,y_test = split.split(split.NP)
    #print feature_map.keys()
    df= df[:,feature_map.keys()]
    #df = df[:, [4, 7, 14, 21, 22, 26, 28]]
    print df.shape, y.shape
    train_model_minC_dict = {}
    for train_model_name in train_model_objects:
        avg_error_list=[]
        print train_model_name
        min_error= sys.maxint
        min_c_val=0
        for c_val in train_model_objects[train_model_name].get_hyper_param("C"):
            print "Current cval: %lf" % c_val
            #df = df.iloc[:, 0:5]
            sec_size = int(len(df)/k)
            error = 0
            param_dict = train_model_objects[train_model_name].fixed_params
            #print param_dict
            param_dict["C"] = c_val
            for i in xrange(k):
                #df_train = pd.concat([df[0:i*sec_size], df[(i+1)*sec_size:k*sec_size]])
                #y_train = pd.concat([y[0:i*sec_size], y[(i+1)*sec_size:k*sec_size]])
                df_train = np.concatenate((df[0:i*sec_size], df[(i+1)*sec_size:k*sec_size]), axis=0)
                y_train = np.concatenate((y[0:i * sec_size], y[(i + 1) * sec_size:k * sec_size]), axis=0)

                trained_model = training_models.training(train_model_name, param_dict, df_train, y_train)
                error += training_models.testing_error(trained_model, X_test=df[i*sec_size:(i+1)*sec_size],
                                              y_test=y[i*sec_size:(i+1)*sec_size])
                #print "Error"+str(error)

            avg_error = error/float(k)
            avg_error_list.append(avg_error)
            print "avg_error: %lf" % avg_error
            if avg_error < min_error:
                min_error = avg_error
                min_c_val = c_val

        print train_model_name + "/ Min_c: %lf, Min Error: %lf" %(min_c_val, min_error)
        train_model_minC_dict[train_model_name] = min_c_val
        plt.plot(train_model_objects[train_model_name].get_hyper_param("C"),avg_error_list,'-o')
        plt.xlabel("C (Hyperparameter)")
        plt.ylabel("Average k-fold cross validation error")
        if not os.path.exists("parameter_tuning"):
            os.makedirs("parameter_tuning")
        plt.savefig("parameter_tuning/"+train_model_name + str(random.randint(0, 100000))+".png")
        plt.clf()
    return train_model_minC_dict


if __name__ == "__main__":
    kfold_cross_validation(5)


