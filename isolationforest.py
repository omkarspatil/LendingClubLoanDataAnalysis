import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib
np.set_printoptions(threshold=np.nan)


rng = np.random.RandomState(42)
#fx = np.loadtxt(fname = "loan_x_default_last.csv", delimiter = ',')
#fy = np.loadtxt(fname = "loan_y_default_last.csv", delimiter = ',')
fx = np.loadtxt(fname = "small_loan_x.csv", delimiter = ',')
fy = np.loadtxt(fname = "small_loan_y.csv", delimiter = ',')

X = fx[10000:19988, [5,11]]
y = fy[10000:19988, 1:]
print X[9987]
#We have all the outlying labels at the end of the dataset and we remove them here manually as we do not wish to train
#on them, we also select the top two columns we obtained from feature selection for training purposes.
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
print X.shape
X_outliers = fx[19988:, [5,11]]
print X_outliers
# fit the model
clf = IsolationForest(max_samples=20000, random_state=rng)
clf.fit(X_train)
print 'Trained'
y_pred_train = clf.predict(X_train)
y_pred_test = clf.predict(X_test)
print y_pred_test
y_pred_outliers = clf.predict(X_outliers)
print y_pred_outliers

# plot the line, the samples, and the nearest vectors to the plane
xx, yy = np.meshgrid(np.linspace(-50, 50, 50), np.linspace(-50, 50, 50))
Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.title("IsolationForest")
plt.contourf(xx, yy, Z, cmap=plt.cm.Blues_r)
plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), 0, 7), cmap=plt.cm.PuBu)
a = plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='darkred')
plt.contourf(xx, yy, Z, levels=[0, Z.max()], colors='palevioletred')

b1 = plt.scatter(X_train[:, 0], X_train[:, 1], c='white',
                 s=20, edgecolor='k')
b2 = plt.scatter(X_test[:, 0], X_test[:, 1], c='green',
                 s=20, edgecolor='k')
c = plt.scatter(X_outliers[:, 0], X_outliers[:, 1], c='red',
                s=20, edgecolor='k')
plt.axis('tight')
#plt.xlim((-5, 5))
#plt.ylim((-5, 5))
plt.legend([a.collections[0],b1, b2, c],
           ["learned frontier","training observations",
            "new regular observations", "new abnormal observations"],
           loc="upper left")
plt.show()


plt.title("One-Class SVM")
plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), 0, 7), cmap=plt.cm.PuBu)
a = plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='darkred')
plt.contourf(xx, yy, Z, levels=[0, Z.max()], colors='palevioletred')

s = 40
b1 = plt.scatter(X_train[:, 0], X_train[:, 1], c='white', s=s, edgecolors='k')
b2 = plt.scatter(X_test[:, 0], X_test[:, 1], c='blueviolet', s=s,
                 edgecolors='k')
c = plt.scatter(X_outliers[:, 0], X_outliers[:, 1], c='gold', s=s,
                edgecolors='k')
plt.axis('tight')
plt.xlim((-50, 50))
plt.ylim((-50, 50))
plt.legend([a.collections[0], b1, b2, c],
           ["learned frontier", "training observations",
            "new regular observations", "new abnormal observations"],
           loc="upper left",
           prop=matplotlib.font_manager.FontProperties(size=11))
plt.show()
