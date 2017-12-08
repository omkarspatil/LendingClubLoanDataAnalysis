import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from recursive_feature_selection import select_k_features
import split
from crossvalidation import kfold_cross_validation
import model_names
def make_meshgrid(x, y):
    """Create a mesh of points to plot in

    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns
    -------
    xx, yy : ndarray
    """
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1

    xx, yy = np.meshgrid(np.arange(x_min, x_max,(x_max-x_min)/1000),np.arange(y_min, y_max,(y_max-y_min)/1000))
    #print xx,yy
    return xx, yy


def plot_contours(ax, clf, xx, yy, **params):
    """Plot the decision boundaries for a classifier.

    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out



# Take the first two features. We could avoid this by using a two-dim dataset
train_model_minC_dict=kfold_cross_validation(5)
X_main, X_test, y_main, y_test = split.split(split.NP)
feature_map = select_k_features(7)
#feature_map={4: 'dti', 10: 'int_rate_updated', 12: 'fico_range_high_updated', 13: 'fico_range_low_updated', 22: 'sub_grade_updated', 25: 'total_acc', 28: 'total_rec_late_fee'};
print feature_map.keys()
for i in xrange(len(feature_map.keys())):
    for j in xrange(i+1, len(feature_map.keys())):
        print i,j

        X = X_main[ :, [feature_map.keys()[i], feature_map.keys()[j]]]

        #y = y_main.T.tolist()[0]
        y= y_main

        print y
        print "Shapes :",X.shape,y.shape
        # we create an instance of SVM and fit out data. We do not scale our
        # data since we want to plot the support vectors
        C = 0.00000001  # SVM regularization parameter


        """models2 = (svm.SVC(kernel='linear', C=0.00014),
                   svm.LinearSVC(C=0.6),
                   svm.SVC(kernel='rbf', C=0.6),
                    svm.SVC(kernel='poly', degree=5, C=C))"""

        models2 = (svm.SVC(kernel='linear', C=train_model_minC_dict[model_names.LINEAR_MODEL]),
                  svm.LinearSVC(C=train_model_minC_dict[model_names.LINEAR_SVC_MODEL]),
                  svm.SVC(kernel='rbf',C=train_model_minC_dict[model_names.RBF_MODEL]))
                  #svm.SVC(kernel='poly', degree=2, C=C))

        models = []
        for index, m in enumerate(models2):
          models.append(m.fit(X,y))

        titles = ('SVC with linear kernel',
                  'LinearSVC (linear kernel)',
                  'SVC with RBF kernel',
                  'Polynomial Kernel Degree 5')

        # Set-up 2x2 grid for plotting.
        fig, sub = plt.subplots(2,2)
        plt.subplots_adjust(wspace=0.4, hspace=0.4)

        X0, X1 = X[:, 0], X[:, 1]
        np.random.rand()
        xx, yy = make_meshgrid(X0, X1)


        count = 0

        for clf, title, ax in zip(models, titles, sub.flatten()):
            plot_contours(ax, clf, xx, yy,cmap=plt.cm.coolwarm, alpha=0.8)
            ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
            ax.set_xlim(xx.min(), xx.max())
            ax.set_ylim(yy.min(), yy.max())
            ax.set_xlabel(feature_map.values()[i])
            ax.set_ylabel(feature_map.values()[j])
            ax.set_xticks(())
            ax.set_yticks(())
            ax.set_title(title)
        plt.savefig("plot_images/"+feature_map.values()[i] + "_VS_" + feature_map.values()[j]+".png")
