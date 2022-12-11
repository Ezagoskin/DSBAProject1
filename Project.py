import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import streamlit as st

st.sidebar.markdown('''
# Sections
- [Data preparing](#data-preparing)
- [Little data clearing](#little-data-clearing)
- [Transformations](#transformations)
- [Data overview](#data-overview)
- [Data comparison](#data-comparison)
- [Last data comparison](#last-data-comparison)
''', unsafe_allow_html=True)

st.title('Analysis of user churn data for a telecom company')
st.header('Data preparing')

st.code("""
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import streamlit as st
""")

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
st.code('''
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 5282 entries, 0 to 5281
Data columns (total 20 columns):
 #   Column                    Non-Null Count  Dtype  
---  ------                    --------------  -----  
 0   ClientPeriod              5282 non-null   int64  
 1   MonthlySpending           5282 non-null   float64
 2   TotalSpent                5282 non-null   object 
 3   Sex                       5282 non-null   object 
 4   IsSeniorCitizen           5282 non-null   int64  
 5   HasPartner                5282 non-null   object 
 6   HasChild                  5282 non-null   object 
 7   HasPhoneService           5282 non-null   object 
 8   HasMultiplePhoneNumbers   5282 non-null   object 
 9   HasInternetService        5282 non-null   object 
 10  HasOnlineSecurityService  5282 non-null   object 
 11  HasOnlineBackup           5282 non-null   object 
 12  HasDeviceProtection       5282 non-null   object 
 13  HasTechSupportAccess      5282 non-null   object 
 14  HasOnlineTV               5282 non-null   object 
 15  HasMovieSubscription      5282 non-null   object 
 16  HasContractPhone          5282 non-null   object 
 17  IsBillingPaperless        5282 non-null   object 
 18  PaymentMethod             5282 non-null   object 
 19  Churn                     5282 non-null   int64  
dtypes: float64(1), int64(3), object(16)
memory usage: 825.4+ KB
''')

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
st.code("""
{1048: ' ',
 1707: ' ',
 2543: ' ',
 3078: ' ',
 3697: ' ',
 4002: ' ',
 4326: ' ',
 4551: ' ',
 4598: ' '}
""")

st.code("""
df.iloc[list(a.keys()), :]
""")
df.iloc[list(a.keys()), :]

st.markdown('All this strings have 0 in ClientPeriod')

st.code("""
sum(df.loc[:, ['ClientPeriod']].values == 0)
""")
st.code("""
array([9])
""")

st.markdown('And they are the only strings which have 0 in ClientPeriod')
st.markdown('It can be concluded that ' ' in TotalSpent means that TotalSpent is equal 0')

st.code("""
# Change ' ' in TotalSpent to 0
df['TotalSpent'][df['TotalSpent'] == ' '] = 0
df['TotalSpent'] = df['TotalSpent'].astype(float)
""")
df['TotalSpent'][df['TotalSpent'] == ' '] = 0
df['TotalSpent'] = df['TotalSpent'].astype(float)

st.subheader('Transformations')

st.code("""
old = df.copy() #We will need it later
""")
old = df.copy()  # We will need it later

st.markdown('Look what values categorical columns have')

st.code("""
for col_name in cat_cols:
    print(col_name)
    print(df[col_name].unique())
    print()
""")
st.code('''
Sex
['Male' 'Female']

IsSeniorCitizen
[0 1]

HasPartner
['Yes' 'No']

HasChild
['Yes' 'No']

HasPhoneService
['Yes' 'No']

HasMultiplePhoneNumbers
['No' 'Yes' 'No phone service']

HasInternetService
['No' 'Fiber optic' 'DSL']

HasOnlineSecurityService
['No internet service' 'No' 'Yes']

HasOnlineBackup
['No internet service' 'No' 'Yes']

HasDeviceProtection
['No internet service' 'No' 'Yes']

HasTechSupportAccess
['No internet service' 'Yes' 'No']

HasOnlineTV
['No internet service' 'No' 'Yes']

HasMovieSubscription
['No internet service' 'No' 'Yes']

HasContractPhone
['One year' 'Two year' 'Month-to-month']

IsBillingPaperless
['No' 'Yes']

PaymentMethod
['Mailed check' 'Credit card (automatic)' 'Electronic check'
 'Bank transfer (automatic)']

''')

st.markdown('Most of categorical columns may be reformatted to int64 without loss of data')

st.code("""
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
""")
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

st.code("""
# Make data easier to analyse and use for machine learning
df[yes_or_no_cols] = (df[yes_or_no_cols] == 'Yes').astype(int)
df['Sex'] = (df['Sex'] == 'Male').astype(int)
""")
df[yes_or_no_cols] = (df[yes_or_no_cols] == 'Yes').astype(int)
df['Sex'] = (df['Sex'] == 'Male').astype(int)

st.markdown('Object parameter HasContractPhone can be changed to int64 ContractDuration')

st.code("""
df['HasContractPhone'][df['HasContractPhone'] == 'Month-to-month'] = 0  
df['HasContractPhone'][df['HasContractPhone'] == 'One year'] = 1
df['HasContractPhone'][df['HasContractPhone'] == 'Two year'] = 2
new_columns = list(df.columns)
new_columns[-4] = 'ContractDuration'
df.columns = new_columns
df['ContractDuration'] = df['ContractDuration'].astype(int)
""")
df['HasContractPhone'][df['HasContractPhone'] == 'Month-to-month'] = 0
df['HasContractPhone'][df['HasContractPhone'] == 'One year'] = 1
df['HasContractPhone'][df['HasContractPhone'] == 'Two year'] = 2
new_columns = list(df.columns)
new_columns[-4] = 'ContractDuration'
df.columns = new_columns
df['ContractDuration'] = df['ContractDuration'].astype(int)

st.markdown('Create an additional parameter showing how many services of the company every customer uses')

st.code("""
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
""")
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

st.code("""
df[['InternetService']] = df[['HasInternetService']]
df[['HasInternetService']] = (df[['HasInternetService']] != 'No').astype(int)
df['NumberOfServices'] = df[serv].sum(axis=1)
old['NumberOfServices'] = df['NumberOfServices']
num_cols.append('NumberOfServices')
""")
df[['InternetService']] = df[['HasInternetService']]
df[['HasInternetService']] = (df[['HasInternetService']] != 'No').astype(int)
df['NumberOfServices'] = df[serv].sum(axis=1)
old['NumberOfServices'] = df['NumberOfServices']
num_cols.append('NumberOfServices')

st.code("""
#Rearrange columns order a bit
cols = df.columns.tolist()
cols = cols[:-3] + cols[-2:]+ cols[-3:-2]
df = df[cols]
df.head()
""")
cols = df.columns.tolist()
cols = cols[:-3] + cols[-2:] + cols[-3:-2]
df = df[cols]
st.dataframe(df.head())

st.code("""
df.info())
""")
st.dataframe(df.info())

st.markdown('Now there is only two object columns. It makes data more convenient for a range of analysis types')
st.header('Data overview')
st.markdown('Start with representing some information about numeric columns')

st.code("""
df[num_cols].describe()
""")
st.dataframe(df[num_cols].describe())

st.code("""
fig, axis = plt.subplots(1, 4, figsize=(20,5))
for i in range(len(num_cols)):
    axis[i].hist(df[num_cols[i]], bins=9)
    axis[i].set_title(num_cols[i])
plt.show()
""")
fig, axis = plt.subplots(1, 4, figsize=(20, 5))
for i in range(len(num_cols)):
    axis[i].hist(df[num_cols[i]], bins=9)
    axis[i].set_title(num_cols[i])
st.pyplot(fig)

st.markdown('Then some information about categorical columns')

st.code("""
fig, ax = plt.subplots(4, 4, figsize=(20, 20))
ax = ax.ravel()
for i, cat_col in enumerate(cat_cols):
    ax[i].set_title(cat_col)
    ax[i].pie(old[cat_col].value_counts(), labels=old[cat_col].value_counts().index)
    ax[i].legend()
fig.show()
""")
fig, ax = plt.subplots(4, 4, figsize=(20, 20))
ax = ax.ravel()
for i, cat_col in enumerate(cat_cols):
    ax[i].set_title(cat_col)
    ax[i].pie(old[cat_col].value_counts(), labels=old[cat_col].value_counts().index)
    ax[i].legend()
st.pyplot(fig)

st.markdown('And a bit about target column')

st.code("""
fig, ax = plt.subplots()
ax.set_title(target_col)
ax.pie(df[target_col].value_counts(), autopct='%1.1f%%', labels=df[target_col].value_counts().index)
ax.legend()
plt.show()
d = dict(df[target_col].value_counts())
print(f'{d[0]} of customers stayed with firm, {d[1]} left')
""")
fig, ax = plt.subplots()
ax.set_title(target_col)
ax.pie(df[target_col].value_counts(), autopct='%1.1f%%', labels=df[target_col].value_counts().index)
ax.legend()
st.pyplot(fig)
d = dict(df[target_col].value_counts())
st.markdown(f'{d[0]} of customers stayed with firm, {d[1]} left')

st.header('Data comparison')
st.markdown('Start with pairplot as it gives some general overview about how columns are connected')

# In[199]:

st.code("""
fig = sns.pairplot(data=df[num_cols+[target_col]],hue=target_col)
""")
fig = sns.pairplot(data=df[num_cols + [target_col]], hue=target_col)
st.pyplot(fig)

st.markdown('''
We can see that in ClientPeriod-MonthlySpending chart blue and orange points are located in different parts, 
so check how relation between ClientPeriod and MonthlySpending depands on whether client left the firm or not
''')

# In[200]:

st.code("""
fig, ax = plt.subplots(1, 2, figsize=(20, 5))
sns.boxplot(data=df[df['Churn']==0],x='ClientPeriod', y='MonthlySpending',ax=ax[0])
ax[0].set_title('Stayed')
sns.boxplot(data=df[df['Churn']==1],x='ClientPeriod', y='MonthlySpending',ax=ax[1])
ax[1].set_title('Left the firm')
""")
fig, ax = plt.subplots(1, 2, figsize=(20, 5))
sns.boxplot(data=df[df['Churn'] == 0], x='ClientPeriod', y='MonthlySpending', ax=ax[0])
ax[0].set_title('Stayed')
sns.boxplot(data=df[df['Churn'] == 1], x='ClientPeriod', y='MonthlySpending', ax=ax[1])
ax[1].set_title('Left the firm')
st.pyplot(fig)

st.markdown('''
It can be concluded that for every value of ClientPeriod MonthlySpendings of those
who left the firm were on average noticeably higher than of those who stayed
''')
st.markdown('''
The next chart shows this pattern even more clear. Orange and blue lines are parallel, 
and orange lies 20 points higher. It is a very valuable result
''')

st.code("""
fig = sns.lmplot(data=df,x='ClientPeriod', y='MonthlySpending',hue='Churn')
""")
fig = sns.lmplot(data=df, x='ClientPeriod', y='MonthlySpending', hue='Churn')
st.pyplot(fig)

st.markdown('Also we see that, in general, the more ClientPeriod, the more MonthlySpendings')
st.markdown('Now, something about correlation')

st.code("""
fig = df.corr().style.background_gradient(cmap='coolwarm').set_precision(2)
fig
""")
fig = df.corr().style.background_gradient(cmap='coolwarm').set_precision(2)
fig

st.markdown('1) TotalSpent strongly correlate both ClientPeriod and MonthlySpending which is not surprizing at all')
st.markdown('2) We can assume that clients with larger ClientPeriod tend to have more long-term contracts')
st.markdown('''
3) It can be noticed that clients with long ClientPeriod or long-term contract are less likely to leave the firm
''')
st.markdown('4) Sex correlate with nothing')
st.markdown('Actually, there is a lot of what can be concluded from this wonderful table, but we are to stop here')
st.markdown('The next charts confirms point 2 and partly confirm point 4')

st.code("""
df1 = df[df['Sex']==1]
df2 = df[df['Sex']==0]
cond = [df['ContractDuration']==0, df['ContractDuration']==1, df['ContractDuration']==2]
titl = ['Month-to-month', 'One year', 'Two years']
""")
df1 = df[df['Sex'] == 1]
df2 = df[df['Sex'] == 0]
cond = [df['ContractDuration'] == 0, df['ContractDuration'] == 1, df['ContractDuration'] == 2]
titl = ['Month-to-month', 'One year', 'Two years']

st.code("""
fig, axis = plt.subplots(1,3, figsize=(20,5))
for j in range(3):
    axis[j].hist([df1[cond[j]]['ClientPeriod'],df2[cond[j]]['ClientPeriod']], bins=9)
    axis[j].legend(['Male','Female'])
    axis[j].set_title(titl[j])
plt.show()
""")
fig, axis = plt.subplots(1, 3, figsize=(20, 5))
for j in range(3):
    axis[j].hist([df1[cond[j]]['ClientPeriod'], df2[cond[j]]['ClientPeriod']], bins=9)
    axis[j].legend(['Male', 'Female'])
    axis[j].set_title(titl[j])
st.pyplot(fig)

st.header('Last data comparison')

st.markdown('''
Churn has negative correlation with parameters HasOnlineSecurityService, HasOnlineBackup, HasDeviceProtection and 
HasTechSupportAccess. It is possible to put forward a **hypothesis** that the more serveces client has, 
the less chance that he leave the firm. Moreover, considering correlation coafficients of Churn with 
IsSeniorCitizen and HasPartner, we may assume that relatevely young married people 
even less likely to leave the firm for every value of Number of serveces
''')
# In[205]:

st.code("""
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
""")
dfss = df[(df['IsSeniorCitizen'] == 1) & (df['HasPartner'] == 0)]
dfsm = df[(df['IsSeniorCitizen'] == 1) & (df['HasPartner'] == 1)]
dfys = df[(df['IsSeniorCitizen'] == 0) & (df['HasPartner'] == 0)]
dfym = df[(df['IsSeniorCitizen'] == 0) & (df['HasPartner'] == 1)]
dss = []
dsm = []
dys = []
dym = []
d = []
for i in range(1, 10):
    dfi = dfss[df['NumberOfServices'] == i]['Churn']
    dss.append(sum(list(dfi)) / len(dfi))
for i in range(1, 10):
    dfi = dfsm[df['NumberOfServices'] == i]['Churn']
    dsm.append(sum(list(dfi)) / len(dfi))
for i in range(1, 10):
    dfi = dfys[df['NumberOfServices'] == i]['Churn']
    dys.append(sum(list(dfi)) / len(dfi))
for i in range(1, 10):
    dfi = dfym[df['NumberOfServices'] == i]['Churn']
    dym.append(sum(list(dfi)) / len(dfi))
for i in range(1, 10):
    dfi = df[df['NumberOfServices'] == i]['Churn']
    d.append(sum(list(dfi)) / len(dfi))

st.code("""
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(range(1, 10), dss)
ax.plot(range(1, 10), dsm)
ax.plot(range(1, 10), dys)
ax.plot(range(1, 10), dym)
ax.plot(range(1, 10), d)
ax.legend(['Senior&Single','Senior&Married','Young&Single','Young&Married','All'])
""")
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(range(1, 10), dss)
ax.plot(range(1, 10), dsm)
ax.plot(range(1, 10), dys)
ax.plot(range(1, 10), dym)
ax.plot(range(1, 10), d)
ax.legend(['Senior&Single', 'Senior&Married', 'Young&Single', 'Young&Married', 'All'])
st.pyplot(fig)

st.subheader('''
The hypotesis was partly confirmd. We knew that function showing ratio of left depending on 
NumberOfServices has form of mountine with pike at 3 purchaced surveces. Young and Married people are actually 
less likely to leave a firm for every value of NumberOfServices
''')
