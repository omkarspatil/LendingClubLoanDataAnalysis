import split
from sklearn.feature_selection import RFE
from sklearn.svm import LinearSVC
from svm_forward_fitting import svm_forward


def select_k_features(k):
    X_train, X_test, y_train, y_test = split.split(split.NP)
    features=["addr_state_updated","annual_inc","application_type_updated","delinq_2yrs","dti","earliest_cr_line_updated","funded_amnt","funded_amnt_inv","grade_updated","home_ownership_updated","int_rate_updated","last_credit_pull_d_updated","fico_range_high_updated","fico_range_low_updated","last_pymnt_amnt","last_pymnt_d_updated","loan_amnt","open_acc","purpose_updated","pymnt_plan_updated","revol_bal","revol_util_updated","sub_grade_updated","verification_status_updated","term_updated","total_acc","total_pymnt","total_pymnt_inv","total_rec_late_fee","acc_now_delinq","tot_coll_amt","tot_cur_bal","emp_length_updated"]

    print len(features)

    clf = LinearSVC()

    #X = X_train.tolist()
    #y = y_train.T.tolist()
    X = X_train
    y = y_train

    selector = RFE(clf, k)
    selector = selector.fit(X, y)

    print selector.support_
    print selector.ranking_

    map_selected_features = {}
    count=0

    for b,feature in zip(selector.support_,features):
        if b:
            map_selected_features[count]=feature
            print feature
        count=count+1

    return map_selected_features

'''
We tried using forward fitting based feature selection with SVM 
y = y_train.T.tolist()
inty = [int(y) for y in y[0]]
clfIn = svm.SVC(kernel="linear",C=0.001)
print inty
list_F = svm_forward(clfIn,X_train, inty,10)
print list_F
print "Forward Fitting: "
for index in list_F:
    print features[index]
'''



