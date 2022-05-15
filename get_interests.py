import user_data
import pandas as pd
import DB_manager as db

def intrests(user_ids):
    print("finding intrests")
    male = 0
    female = 0

    interest_list = pd.Series()
    for user in user_ids:
        gender, interests = user_data.main(user)
        if gender == "male":
            male += 1
        elif gender == "female":
            female += 1
        else:
            continue

        for list in interests:
            interest_list = pd.concat([interest_list,pd.Series(list)], ignore_index=True)

    print("percentafe of females: {f}".format(f = (female/(male+female))))
    print(interest_list)
    with open('genders.csv','w') as fd:
        fd.write("{f} {m}".format(f=female,m=male))
    counts = interest_list.value_counts()
    counts.to_csv('counts.csv')

 

def get_user_ids():
    records = db.query("SELECT * FROM Tweets WHERE rowid >=0 AND rowid <= 100")
    print(records)
    return records['UserID']

def main():
    users = get_user_ids()
    intrests(users)

# calling the main function
if __name__ ==  "__main__":
    main()
