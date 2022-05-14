import pandas as pd
import sqlite3
from os import path
import Tweets as Tweets
from tqdm import tqdm

# DB MANAGER
# all DB functions are seperated so that they can be used by other scripts
# If using the DB make sure to close the connection to stop errors

def main():
    #Get tweets from Tweets.py
    data = Tweets.main()
    con,cursor = connect_to_db()
    Store_tweets(data,cursor,con)
    
    #close connection to db
    close_connection(con)

def connect_to_db():
    # connect to Databse
    con = sqlite3.connect('Main.db')
    cursor = con.cursor()

    #Check to see if table already exists
    try:
        cursor.execute('''SELECT 1 FROM Tweets;''')
    except:
        print('DB NOT FOUND RUNNING CONFIG')
        config_db(cursor)


    return con,cursor

def close_connection(con):
    #close connection to db
    con.close()

def Store_tweets(data,cursor,con):
    data = data.drop(['scores','TweetID'], axis=1)
    print("Inserting Records into DB")
    for i in tqdm(range(0,int(data.size/7)), desc='Inserting...'):
        Values = data.loc[i].tolist()
        Values[3] =  str(Values[3]) #converts the records values into correct data type for SQL
        Values[0] = int(Values[0])
        Insert_record(cursor,Values,con)
    print("Records Sucessfully Inserted")


def Insert_record(cursor,Values,con):
    SQL = '''INSERT INTO Tweets(PK,User_ID,Name,User_location,Date,Text,Compound,Comp_score) VALUES(NULL,?,?,?,?,?,?,?) '''

    try:
        cursor.execute(SQL, Values)
        con.commit()
    except Exception as e:
        print("ERROR when inserting Value: {}".format(e))
        return False

    return True


def config_db(cursor):
    print('__ CONFIGURING DATABASE __')

    cursor.execute('''CREATE TABLE Tweets
                    ( PK INTEGER PRIMARY KEY,
                    User_ID INTEGER,
                    Name TEXT,
                    User_location TEXT,
                    Date TEXT,
                    Text Text,
                    Compound REAL,
                    Comp_score REAL )''')

    print("DB SUCESSFULLY CONFIGURED")
    
def query(sql_query): #rowid starts at 1 for records
    #gets records from start to end in the same panada format they came
    con,cursor = connect_to_db()
    try:
        df = pd.DataFrame(columns = ['PK','UserID','Name','User Location','Date and Time','Text','compound','comp_score'])
        rows = cursor.execute(sql_query)
        
        for row in rows:
            p_row = pd.DataFrame([row], columns = ['PK','UserID','Name','User Location','Date and Time','Text','compound','comp_score'])
            df = pd.concat([df,p_row],ignore_index=True)
        
        #print(df)

    except Exception as e:
        print("ERROR WHILE RETRIEVING RECORDS :")
        print(e)

    close_connection(con)

    return df

# calling the main function
if __name__ ==  "__main__":
    start = 9750
    end = 10000
    #query("SELECT * FROM Tweets WHERE rowid >= {s} AND rowid <= {e}".format(s = start,e = end))
    main()
