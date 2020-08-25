#!/usr/bin/env python
# coding: utf-8

# In[25]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
import matplotlib.ticker as plticker
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


# In[26]:


ucl_data = pd.read_csv('ucl_data.csv')
ucl_data.head()


# In[27]:


winner = []
for i in range (len(ucl_data['Team'])):
    if ucl_data ['Goalf'][i] > ucl_data['Goala'][i]:
        winner.append(ucl_data['Team'][i])
    else:
        winner.append('other')
ucl_data['winning'] = winner


ucl_data.head()


# In[28]:


Homeg = []
Away = []
for i in range (len(ucl_data['Team'])):
    if ucl_data ['Home'][i] == 1:
        Homeg.append(ucl_data['Team'][i])
        Away.append('Other')
    else:
        Homeg.append('Other')
        Away.append(ucl_data['Team'][i])
ucl_data['Homeg'] = Homeg
ucl_data['Away'] = Away

ucl_data.head()


# In[29]:


matches_data = ucl_data
matches_data


# In[30]:


matches_data = matches_data.reset_index(drop=True)
matches_data.loc[matches_data.winning == matches_data.Homeg,'winning']=2
matches_data.loc[matches_data.winning == matches_data.Away,'winning']=1
matches_data.loc[matches_data.winning == 'other','winning']=0


matches_data


# In[31]:



matches_data = matches_data.drop(['Team'], axis=1)
final = pd.get_dummies(matches_data, prefix=['Homeg', 'Away'], columns=['Homeg', 'Away'])


X = final.drop(['winning'], axis=1)
y = final["winning"]
y = y.astype('int')


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)


# In[32]:


final.head()


# In[33]:


result_model = LogisticRegression()
result_model.fit(X_train, y_train)
score = result_model.score(X_train, y_train)
score2 = result_model.score(X_test, y_test)

print("Training set accuracy: ", '%.3f'%(score))
print("Test set accuracy: ", '%.3f'%(score2))


# In[34]:


ranking = pd.read_csv('ranking.csv') 
Rounds = pd.read_csv('Rounds.csv')


pred_set = []


# In[35]:


Rounds.insert(1, 'first_position', Rounds['Homeg'].map(ranking.set_index('Team')['position']))
Rounds.insert(2, 'second_position', Rounds['Away'].map(ranking.set_index('Team')['position']))

Rounds.tail()


# In[36]:


for index, row in Rounds.iterrows():
    if row['first_position'] < row['second_position']:
        pred_set.append({'Homeg': row['Homeg'], 'Away': row['Away'], 'winning': None})
    else:
        pred_set.append({'Homeg': row['Away'], 'Away': row['Homeg'], 'winning': None})
        
pred_set = pd.DataFrame(pred_set)
backup_pred_set = pred_set

pred_set.head()


# In[37]:


pred_set = pd.get_dummies(pred_set, prefix=['Homeg', 'Away'], columns=['Homeg', 'Away'])


missing_cols = set(final.columns) - set(pred_set.columns)
for c in missing_cols:
    pred_set[c] = 0
pred_set = pred_set[final.columns]

pred_set = pred_set.drop(['winning'], axis=1)

pred_set.head()


# In[38]:


predictions = result_model.predict(pred_set)
count = 1
for i in range(Rounds.shape[0]):
    num = 1
    print("Match" +str(count)+ ": " + backup_pred_set.iloc[i, 1] + " VS " + backup_pred_set.iloc[i, 0])
    if predictions[i] == 1:
        print("Winner: " + backup_pred_set.iloc[i, 1])
    elif predictions[i] == 0:
       # print("Penalties")
        if '%.3f'%(result_model.predict_proba(pred_set)[i][2]) > '%.3f'%(result_model.predict_proba(pred_set)[i][0]):
            print( backup_pred_set.iloc[i, 1] + ' winning: ', '%.3f'%(result_model.predict_proba(pred_set)[i][2]))
        else:
            print( "Winner: "+backup_pred_set.iloc[i, 0] )
        num = 0 
    elif predictions[i] == 2:
        print("Winner: " + backup_pred_set.iloc[i, 0])
    print("")
    count = count+1


# In[ ]:




