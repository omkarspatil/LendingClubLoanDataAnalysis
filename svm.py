from sklearn import svm
import split

X_train, X_test, y_train, y_test = split.split()
clf = svm.SVC()
X = X_train.tolist()
y = y_train.T.tolist()
#print X
#print len(y[0])

clf = svm.SVC(kernel = 'linear', C = 1.0)
clf.fit(X, y[0])

#print clf.predict(X_test.tolist())
#print y_test.T.tolist()[0]

sum=0
for x,y in zip(y[0],y_test.T.tolist()[0]):
    if x-y==0:
        sum+=1

print sum, len(y[0]), sum/float(len(y[0]))