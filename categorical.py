from sklearn.preprocessing import LabelEncoder
import pandas as pd
import math
import numpy as np

# Read in the CSV file and convert "?" to NaN


def toNumericFromMonth(month):
    if type(month) is not float:
        months=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
        monthsval1,monthsval2 = month.split("-")
        ans = months.index(monthsval1)+int(monthsval2)*12
        return ans
    else:
        return 0

def removeMonths(month):
    return str(month).split(" ")[0]

def removePercentage(percentage):
    return float(str(percentage).split("%")[0])

def extractNums(mixed):
    return filter(str.isdigit, str(mixed))


def ws_to_zero(maybe_ws):
    if isinstance(maybe_ws, str):
        try:
            maybe_ws = float(maybe_ws)
        except:
            return 0
    if isinstance(maybe_ws, float) and np.isnan(maybe_ws):
        return 0
    return maybe_ws

df = pd.read_csv("/Users/omkar/Downloads/merged.csv")
df.head()


df = df[(df["loan_status"]=="Default") | (df["loan_status"]=="Charged Off") | (df["loan_status"]=="Fully Paid")]


categorical_columns=["addr_state","application_type","purpose",
                     "grade","home_ownership","sub_grade","fico_range_high",
                     "fico_range_low","pymnt_plan","verification_status","loan_status"]

month_columns = ["earliest_cr_line","last_credit_pull_d","last_pymnt_d"]

percentage_columns=["int_rate","revol_util"]

df['term_updated'] = df['term'].apply(lambda x : removeMonths(x))
df['emp_length_updated'] = df['emp_length'].apply(lambda x : extractNums(x))



for val in percentage_columns:
    df[val+'_updated'] = df[val].apply(lambda x: removePercentage(x))

for month in month_columns:
    print "Encoding " + month
    df[month+'_updated'] = df[month].apply(lambda x : toNumericFromMonth(x))


for category in categorical_columns:
    print "Encoding "+category
    lb_make = LabelEncoder()
    df[category+"_updated"] = lb_make.fit_transform(df[category])
    df[category + '_updated'] +=1



df_updated = df[["addr_state_updated","annual_inc","application_type_updated","delinq_2yrs","dti","earliest_cr_line_updated","funded_amnt","funded_amnt_inv","grade_updated","home_ownership_updated","int_rate_updated","last_credit_pull_d_updated","fico_range_high_updated","fico_range_low_updated","last_pymnt_amnt","last_pymnt_d_updated","loan_amnt","open_acc","purpose_updated","pymnt_plan_updated","revol_bal","revol_util_updated","sub_grade_updated","verification_status_updated","term_updated","total_acc","total_pymnt","total_pymnt_inv","total_rec_late_fee","acc_now_delinq","tot_coll_amt","tot_cur_bal","emp_length_updated"]]


df_updated_y=df[["loan_status_updated"]]

df_updated=df_updated.applymap(lambda x: ws_to_zero(x))


df_updated.to_csv("loan_x_final.csv")
df_updated_y.to_csv("loan_y_final.csv");





