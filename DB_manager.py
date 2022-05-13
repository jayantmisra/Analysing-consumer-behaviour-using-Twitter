import pandas
import sqlite3
from os import path
import Tweets as Tweets

# DB MANAGER
# all DB functions are seperated so that they can be used by other scripts
# If using the DB make sure to close the connection to stop errors


def main():
    # Get tweets from Tweets.py
    data = Tweets.main()
    con, cursor = connect_to_db()
    Store_tweets(data, cursor, con)

    # close connection to db
    close_connection(con)


def connect_to_db():
    # connect to Databse
    con = sqlite3.connect('Main.db')
    cursor = con.cursor()

    # Check to see if table already exists
    try:
        cursor.execute('''SELECT 1 FROM Tweets;''')
    except:
        print('DB NOT FOUND RUNNING CONFIG')
        config_db(cursor)

    return con, cursor


def close_connection(con):
    # close connection to db
    con.close()


def Store_tweets(data, cursor, con):

    print(data.iloc[0])
    data = data.drop(['scores', 'TweetID'], axis=1)
    for i in range(0, int(data.size/7)):
        Values = data.loc[i].tolist()
        Values[3] = str(Values[3])
        Values[0] = int(Values[0])
        Insert_record(cursor, Values, con)


def Insert_record(cursor, Values, con):
    print('Inserting Record')
    SQL = '''INSERT INTO Tweets(PK,User_ID,Name,User_location,Date,Text,Compound,Comp_score) VALUES(NULL,?,?,?,?,?,?,?) '''

    try:
        cursor.execute(SQL, Values)
        con.commit()
    except Exception as e:
        print("ERROR when inserting Value: {}".format(e))
        return False

    print('Record Successfully Inserted')
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


# calling the main function
if __name__ == "__main__":
    main()
