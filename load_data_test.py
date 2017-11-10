from tabulate import tabulate
import sqlite3
#url = "/Users/rahul/Downloads/lending-club-loan-data/database.sqlite"
url_x = "/Users/rahul/Downloads/loan_x_final.csv"
url_y = "/Users/rahul/Downloads/loan_y_final.csv"

f_x, f_y = open(url_x, 'r'), open(url_y, 'r')
f_x1, f_y1 = open(url_x+'_modified', 'w'), open(url_y+'_modified', 'w')

lines_x, lines_y = f_x.readlines(), f_y.readlines()
print len(lines_x), len(lines_y)
print lines_x[0]
print lines_y[0]
default_row = []
for i, xy in enumerate(zip(lines_x, lines_y)):
    #print xy[1].strip('\n')
    if xy[1].split(',')[1].strip('\n') == '2':
        default_row.append(i)
        print xy[1]
    else:
        f_x1.write(xy[0])
        f_y1.write(xy[1])
    # if i == 10:
    #     break
for i in default_row:
    f_x1.write(lines_x[i])
    f_y1.write(lines_y[i])
print default_row
f_x.close()
f_y.close()
f_x1.close()
f_y1.close()

#print lines[1].index('fico_range_high')
# lines[1] = [x.strip('\"') for x in lines[1].split(',')]
# for i,x in enumerate(lines[1]):
#     print i,x
# for l in lines[1:30]:
#     #print l
#     l = [x.strip('\"') for x in l.split(',')]
#     print l[28]
# f.close()
# file_names = ["LoanStats3a_securev1.csv", "LoanStats3b_securev1.csv",
#               "LoanStats3c_securev1.csv", "LoanStats3d_securev1.csv"]
#file_names = ["test1","test2"]
# count = 0
# wf = open("merged.csv", 'w')
# with open("../"+file_names[0], 'r') as rf:
#     lines = rf.readlines()
#     for l in lines[1:]:
#         count += 1
#         wf.write(l)
# for f in file_names[1:]:
#     x = "../"+f
#     with open(x, 'r') as rf:
#         lines = rf.readlines()
#         for line in lines[2:]:
#             count += 1
#             wf.write(line)
# print count
# wf.close()

#print f.readlines()[1].split(',')
# load the CSV file as a numpy matrix
# conn = sqlite3.connect(url)
# #query = '''select count(*) from INFORMATION_SCHEMA.COLUMNS
# # WHERE table_name = loan'''
# #query = '''select * from loan where dti_joint is NULL limit 10'''
#
# # Creating dataframe from query.Teams = conn.execute(query).fetchall()
# Teams = conn.execute(query).fetchall()
# print tabulate(Teams, headers='keys', tablefmt='psql')
