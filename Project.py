import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import streamlit as st
sns.set_theme()

st.title('sth')
st.header('sth')
st.subheader('sth')
st.markdown('sth')
st.code("""
""")

st.title('Analysis of user churn data for a telecom company')
st.header('Data preparing')

st.code("""
df = pd.read_csv("train.csv")
df.head()
""")
df = pd.read_csv("train.csv")
st.dataframe(df.head())

st.code("""
num_cols = [
    'ClientPeriod',
    'MonthlySpending',
    'TotalSpent'
]

cat_cols = [
    'Sex',
    'IsSeniorCitizen',
    'HasPartner',
    'HasChild',
    'HasPhoneService',
    'HasMultiplePhoneNumbers',
    'HasInternetService',
    'HasOnlineSecurityService',
    'HasOnlineBackup',
    'HasDeviceProtection',
    'HasTechSupportAccess',
    'HasOnlineTV',
    'HasMovieSubscription',
    'HasContractPhone',
    'IsBillingPaperless',
    'PaymentMethod'
]

feature_cols = num_cols + cat_cols
target_col = 'Churn'
""")
num_cols = [
    'ClientPeriod',
    'MonthlySpending',
    'TotalSpent'
]
cat_cols = [
    'Sex',
    'IsSeniorCitizen',
    'HasPartner',
    'HasChild',
    'HasPhoneService',
    'HasMultiplePhoneNumbers',
    'HasInternetService',
    'HasOnlineSecurityService',
    'HasOnlineBackup',
    'HasDeviceProtection',
    'HasTechSupportAccess',
    'HasOnlineTV',
    'HasMovieSubscription',
    'HasContractPhone',
    'IsBillingPaperless',
    'PaymentMethod'
]
feature_cols = num_cols + cat_cols
target_col = 'Churn'

st.code("""
df.info()
""")
st.dataframe(df.info())

st.markdown('No null values. But TotalSpent has type object which is strange')
st.subheader('Little data clearing')

st.code("""
# Find not numbers in TotalSpent
a = {}
for i in range(5282):
    try:
        x = float(df['TotalSpent'][i])
    except:
        a[i] = df['TotalSpent'][i]
a
""")
a = {}
for i in range(5282):
    try:
        x = float(df['TotalSpent'][i])
    except:
        a[i] = df['TotalSpent'][i]
a

st.code("""
df.iloc[list(a.keys()), :]
""")
df.iloc[list(a.keys()), :]

st.markdown('All this strings have 0 in ClientPeriod')

st.code("""
sum(df.loc[:, ['ClientPeriod']].values == 0)
""")
print(sum(df.loc[:, ['ClientPeriod']].values == 0))

st.markdown('And they are the only strings which have 0 in ClientPeriod')
st.markdown('It can be concluded that ' ' in TotalSpent means that TotalSpent is equal 0')

# In[185]:


# Change ' ' in TotalSpent to 0
df['TotalSpent'][df['TotalSpent'] == ' '] = 0
df['TotalSpent'] = df['TotalSpent'].astype(float)


# #### Transformations

# In[186]:


old = df.copy() #We will need it later


# Look what values categorical columns have

# In[187]:


for col_name in cat_cols:
    print(col_name)
    print(df[col_name].unique())
    print()


# Most of categorical columns may be reformatted to int64 without loss of data

# In[188]:


yes_or_no_cols = [
    'HasPartner',
    'HasChild',
    'HasPhoneService',
    'HasMultiplePhoneNumbers',
    'HasOnlineSecurityService',
    'HasOnlineBackup',
    'HasDeviceProtection',
    'HasTechSupportAccess',
    'HasOnlineTV',
    'HasMovieSubscription',
    'IsBillingPaperless',
]


# In[189]:


# Make data easier to analyse and use for machine learning
df[yes_or_no_cols] = (df[yes_or_no_cols] == 'Yes').astype(int)
df['Sex'] = (df['Sex'] == 'Male').astype(int)


# Object parameter HasContractPhone can be changed to int64 ContractDuration

# In[190]:


df['HasContractPhone'][df['HasContractPhone'] == 'Month-to-month'] = 0  
df['HasContractPhone'][df['HasContractPhone'] == 'One year'] = 1
df['HasContractPhone'][df['HasContractPhone'] == 'Two year'] = 2
new_columns = list(df.columns)
new_columns[-4] = 'ContractDuration'
df.columns = new_columns
df['ContractDuration'] = df['ContractDuration'].astype(int)


# In[190]:





# Let's create an additional parameter showing how many services of the company every customer uses

# In[191]:


serv = [
    'HasPhoneService',
    'HasMultiplePhoneNumbers',
    'HasOnlineSecurityService',
    'HasOnlineBackup',
    'HasDeviceProtection',
    'HasTechSupportAccess',
    'HasOnlineTV',
    'HasMovieSubscription',
    'HasInternetService',
]


# In[192]:


df[['InternetService']] = df[['HasInternetService']]
df[['HasInternetService']] = (df[['HasInternetService']] != 'No').astype(int)
df['NumberOfServices'] = df[serv].sum(axis=1)
old['NumberOfServices'] = df['NumberOfServices']
num_cols.append('NumberOfServices')


# In[193]:


#Rearrange columns order a bit
cols = df.columns.tolist()
cols = cols[:-3] + cols[-2:]+ cols[-3:-2]
df = df[cols]
df.head()


# In[194]:


df.info()


# Now there is only two object columns. It makes data more convenient for a range of analysis types

# ## Data overview

# Start with representing some information about numeric columns

# In[195]:


df[num_cols].describe()


# In[196]:


fig, axis = plt.subplots(1, 4, figsize=(20,5))
for i in range(len(num_cols)):
    axis[i].hist(df[num_cols[i]], bins=9)
    axis[i].set_title(num_cols[i])
plt.show()


# Then some information about categorical columns

# In[197]:



fig, ax = plt.subplots(4, 4, figsize=(20, 20))
ax = ax.ravel()
for i, cat_col in enumerate(cat_cols):
    ax[i].set_title(cat_col)
    ax[i].pie(old[cat_col].value_counts(), labels=old[cat_col].value_counts().index)
    ax[i].legend()
fig.show()


# And a bit about target column

# In[198]:


fig, ax = plt.subplots()
ax.set_title(target_col)
ax.pie(df[target_col].value_counts(), autopct='%1.1f%%', labels=df[target_col].value_counts().index)
ax.legend()
plt.show()
d = dict(df[target_col].value_counts())
print(f'{d[0]} of customers stayed with firm, {d[1]} left')


# ## Data comparison

# Start with pairplot as it gives some general overview about how columns are connected

# In[199]:


f = sns.pairplot(data=df[num_cols+[target_col]],hue=target_col)


# We can see that in ClientPeriod-MonthlySpending chart blue and orange points are located in different parts, so let's check how relation between ClientPeriod and MonthlySpending depands on whether client left the firm or not

# In[200]:


fig, ax = plt.subplots(1, 2, figsize=(20, 5))
sns.boxplot(data=df[df['Churn']==0],x='ClientPeriod', y='MonthlySpending',ax=ax[0])
ax[0].set_title('Stayed')
sns.boxplot(data=df[df['Churn']==1],x='ClientPeriod', y='MonthlySpending',ax=ax[1])
ax[1].set_title('Left the firm')


# It can be concluded that for every value of ClientPeriod MonthlySpendings of those who left the firm were on average noticeably higher than of those who stayed

# The next chart shows this pattern even more clear. Orange and blue lines are parallel, and orange lies 20 points higher. It is a very valuable result

# In[201]:


sns.lmplot(data=df,x='ClientPeriod', y='MonthlySpending',hue='Churn')


# Also we see that, in general, the more ClientPeriod, the more MonthlySpendings
# 

# Now, something about correlation

# In[202]:


f = df.corr().style.background_gradient(cmap='coolwarm').set_precision(2)
f


# 1) TotalSpent strongly correlate both ClientPeriod and MonthlySpending which is not surprizing at all
# 

# 2) We can assume that clients with larger ClientPeriod tend to have more long-term contracts

# 3) It can be noticed that clients with long ClientPeriod or long-term contract are less likely to leave the firm
# 

# 4) Sex correlate with nothing
# 

# Actually, there is a lot of what can be concluded from this wonderful table, but let's stop here

# The next charts confirms point 2 and partly confirm point 4

# In[203]:


df1 = df[df['Sex']==1]
df2 = df[df['Sex']==0]
cond = [df['ContractDuration']==0, df['ContractDuration']==1, df['ContractDuration']==2]
titl = ['Month-to-month', 'One year', 'Two years']


# In[204]:


fig, axis = plt.subplots(1,3, figsize=(20,5))
for j in range(3):
    axis[j].hist([df1[cond[j]]['ClientPeriod'],df2[cond[j]]['ClientPeriod']], bins=9)
    axis[j].legend(['Male','Female'])
    axis[j].set_title(titl[j])
plt.show()


# ## Last data comparison

# >Churn has negative correlation with parameters HasOnlineSecurityService, HasOnlineBackup, HasDeviceProtection and HasTechSupportAccess. It is possible to put forward a **hypothesis** that the more serveces client has, the less chance that he leave the firm. Moreover, considering correlation coafficients of Churn with IsSeniorCitizen and HasPartner, we may assume that relatevely young married people even less likely to leave the firm for every value of Number of serveces

# In[205]:


dfss = df[(df['IsSeniorCitizen'] == 1) & (df['HasPartner'] == 0)]
dfsm = df[(df['IsSeniorCitizen'] == 1) & (df['HasPartner'] == 1)]
dfys = df[(df['IsSeniorCitizen'] == 0) & (df['HasPartner'] == 0)]
dfym = df[(df['IsSeniorCitizen'] == 0) & (df['HasPartner'] == 1)]

dss= []
dsm = []
dys= []
dym = []
d = []
for i in range(1,10):
    dfi = dfss[df['NumberOfServices'] == i]['Churn']
    dss.append(sum(list(dfi))/len(dfi))
for i in range(1,10):
    dfi = dfsm[df['NumberOfServices'] == i]['Churn']
    dsm.append(sum(list(dfi))/len(dfi))
for i in range(1,10):
    dfi = dfys[df['NumberOfServices'] == i]['Churn']
    dys.append(sum(list(dfi))/len(dfi))
for i in range(1,10):
    dfi = dfym[df['NumberOfServices'] == i]['Churn']
    dym.append(sum(list(dfi))/len(dfi))
for i in range(1,10):
    dfi = df[df['NumberOfServices'] == i]['Churn']
    d.append(sum(list(dfi))/len(dfi))


# In[206]:


l1 = plt.plot(range(1,10), dss)
l2 = plt.plot(range(1,10), dsm)
l3 = plt.plot(range(1,10), dys)
l4 = plt.plot(range(1,10), dym)
l5 = plt.plot(range(1,10), d)
plt.legend(['Senior&Single','Senior&Married','Young&Single','Young&Married','All'])


# The hypotesis was partly confirmd. We knew that function showing ratio of left depending on NumberOfServices has form of mountine with pike at 3 purchaced surveces. Young and Married people are actually less likely to leave a firm for every value of NumberOfServices
