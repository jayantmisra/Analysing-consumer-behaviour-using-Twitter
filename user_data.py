import user_data
import pandas as pd
import DB_manager as db

def intrests(user_ids):
    print("finding intrests")
    male = 0
    female = 0

    interest_list = pd.Series([],dtype ='float64')
    for user in user_ids:
        gender, interests = user_data.main(user)
        if gender == "male":
            male += 1
        elif gender == "female":
            female += 1
        else:
            continue

        interest_list = pd.concat([interest_list,pd.Series(interests)], ignore_index=True)

    #print(interest_list)
    with open('genders.csv','w') as fd:
        fd.write("{f} {m}".format(f=female,m=male))

    for index in range(0,interest_list.size):
        if interest_list[index] == "url" or interest_list[index] == "rt" or interest_list[index] == "uber":
            interest_list.drop([index],inplace=True)
    
    counts = interest_list.value_counts()
    print(counts)
    """for index in range(0,counts.size):
        print(index)
        print(counts[index])
        counts.drop(['url','rt'],inplace = True)
        if counts[index] == "url" or counts[index] == "rt":
            counts.drop([index],inplace = True)"""

    print(counts)
    counts.to_csv('counts.csv')

 

def get_user_ids():
    records = db.query("SELECT * FROM Tweets WHERE rowid >=11000 AND rowid <= 12000")
    print(records)
    return records['UserID']

def main():
    users = get_user_ids()
    intrests(users)

# calling the main function
if __name__ ==  "__main__":
    main()
