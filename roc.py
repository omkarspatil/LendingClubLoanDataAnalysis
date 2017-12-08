import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
from sklearn.metrics import precision_recall_fscore_support
import split
import model_names


def get_result(model,name):
    X_train, X_test, y_train, y_test = split.split(split.NP)
    model.fit(X_train,y_train)
    error_sum=0
    for i in xrange(len(X_test)):
        # print X_test[i]
        # print y_test[i:i+1].iat[0,0]
        # print model.predict(X_test[i:i+1])[0]
        # print y_test.iloc[i, 0:]
        # error_sum += (y_test[i:i+1].iat[0,0]-model.predict(X_test[i:i+1])[0])** 2
        error_sum += diff_error(y_test[i], model.predict(np.reshape(X_test[i, :], (-1, X_test.shape[1])))[0])
    # print "Error sum",error_sum
    y_predict = model.predict(X_test)
    scores= precision_recall_fscore_support(y_test, y_predict, average='macro')
    print name + " Results  :"
    print " Precision :" + scores[0]
    print " Recall :" + scores[1]
    print " F-measure :" + scores[2]
    print " Accuracy : " + str((1-error_sum / float(len(X_test)))*100)


def diff_error(y, pred):
    return 1 if y != pred else 0


def plot_ROC(model,name):
    # shuffle and split training and test sets
    X_train, X_test, y_train, y_test = split.split(split.NP)
    print len(X_train),len(X_test),len(y_train),len(y_test)
    y_train = label_binarize(y_train, classes=[1, 2, 3])
    y_test = label_binarize(y_test, classes=[1, 2, 3])
    print y_train
    print y_test
    n_classes = y_train.shape[1]
    print n_classes

    # Learn to predict each class against the other
    classifier = OneVsRestClassifier(model)
    y_score = classifier.fit(X_train, y_train).decision_function(X_test)

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=1)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=1)

    lw = 2
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title("ROC for" + name)
    plt.legend(loc="lower right")
    plt.savefig(name+"ROC.png")

if __name__ == "__main__":
    models2 = ((svm.SVC(kernel='linear', C=0.00014),model_names.LINEAR_MODEL),
               (svm.LinearSVC(C=0.6),model_names.LINEAR_SVC_MODEL),
               (svm.SVC(kernel='rbf', C=0.6),model_names.RBF_MODEL))

    for index, m in enumerate(models2):
        plot_ROC(m[0],m[1])
        get_result(m[0],m[1])