from tabulate import tabulate
import sqlite3
# URL for the Pima Indians Diabetes dataset (UCI Machine Learning Repository)
url = "/Users/omkar/Downloads/lending-club-loan-data/database.sqlite"
# load the CSV file as a numpy matrix
conn = sqlite3.connect(url)
query = '''select count(distinct(id)) as count from loan'''

# Creating dataframe from query.Teams = conn.execute(query).fetchall()
Teams = conn.execute(query).fetchall()
print tabulate(Teams, headers='keys', tablefmt='psql')
