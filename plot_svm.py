import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from recursive_feature_slection import select_k_features
import copy

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
X_main,y_main,feature_map,selector = select_k_features(7)
print feature_map.keys()
for i in xrange(len(feature_map.keys())):
    for j in xrange(i+1, len(feature_map.keys())):
        print i,j
        #print X[:,[feature_map.keys()[i]]]
        #print X[:,[feature_map.keys()[j]]]

        X = X_main[ :, [feature_map.keys()[i], feature_map.keys()[j]]]
        #print X.shape
        y = y_main.T.tolist()[0]

        ''''''
        # we create an instance of SVM and fit out data. We do not scale our
        # data since we want to plot the support vectors
        C = 0.00000001  # SVM regularization parameter
        models2 = (svm.SVC(kernel='linear', C=C),
                  svm.LinearSVC(C=C),
                  svm.SVC(kernel='rbf'),
                  svm.SVC(kernel='poly', degree=2, C=C))

        models = []
        for index, m in enumerate(models2):
          #print i
          models.append(m.fit(X,y))
          #print i
        #models = (clf.fit(X, y) for clf in models)

        #models = (None, selector, None, None)
        #print "Here"
        # title for the plots
        titles = ('SVC with linear kernel',
                'LinearSVC (linear kernel)',
                  'SVC with RBF kernel',
                  'Polynomial Kernel Degree 2')

        # Set-up 2x2 grid for plotting.
        fig, sub = plt.subplots(2, 2)
        plt.subplots_adjust(wspace=0.4, hspace=0.4)

        #print id(X)
        X0, X1 = X[:, 0], X[:, 1]

        np.random.rand()
        xx, yy = make_meshgrid(X0, X1)
        #print xx
        #print yy
        #print "Done"
        #print len(models)

        #print "Done zip"
        count = 0

        for clf, title, ax in zip(models, titles, sub.flatten()):
            #count += 1
            #if count != 2 or clf is None:
            #    continue
            #print ax
            #print "Set Axis" + title
            plot_contours(ax, clf, xx, yy,cmap=plt.cm.coolwarm, alpha=0.8)
            #print "Set Axis1" + title
            ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
            #print "Set Axis" + title
            ax.set_xlim(xx.min(), xx.max())
            ax.set_ylim(yy.min(), yy.max())

            ax.set_xlabel(feature_map.values()[i])
            ax.set_ylabel(feature_map.values()[j])
            ax.set_xticks(())
            ax.set_yticks(())
            ax.set_title(title)
            #print "Done with "+title

        plt.savefig("plot_images/"+feature_map.values()[i] + "_VS_" + feature_map.values()[j]+".png")
