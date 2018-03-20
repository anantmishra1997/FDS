
# coding: utf-8

# In[9]:


import pandas as pd
import numpy as np
from sklearn import preprocessing, cross_validation, neighbors ,svm

df = pd.read_csv('C:\\Users\\Anant\\PycharmProjects\\Rajasthan Hackathon 4.0\\FDS\\dashboard\\notebook\\bank.csv', delimiter=";")

df = df[['age', 'job', 'marital', 'education', 'balance','housing', 'loan', 'duration','campaign','pdays','previous', 'poutcome','y']]

def handle_non_numerical_data(df):
    columns = df.columns.values
    
    for column in columns:
        text_digit_vals = {}
        def convert_to_int(val):
            return text_digit_vals[val]
        
        if df[column].dtype != np.int64 and df[column].dtype != np.float64:
            column_contents = df[column].values.tolist()
            unique_elements = set(column_contents)
            x = 1
            for unique in unique_elements:
                if unique not in text_digit_vals:
                    text_digit_vals[unique] = x
                    x+=1
            df[column] = list(map(convert_to_int, df[column]))
    
    return df

df = handle_non_numerical_data(df)

def bank_model():
    global df
    y = df[["y"]]
    x = df.drop(['y'],1)
    x = x.values
    y = y.values
    x_train, x_test, y_train, y_test = cross_validation.train_test_split(x,y, test_size=0.2)
    clf = svm.SVC()
    clf.fit(x_train, y_train)
    accuracy = clf.score(x_test, y_test)
    print(accuracy)
    return clf

# c=clf.predict([[31,1,1,2,406,2,2,736,1,-1,0,3]])
# print(c)


# In[8]:




