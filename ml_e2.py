print("Perform Data Preprocessing like outlier detection, handling missing value, analyzing redundancy and normalization on Bank Marketing Dataset")
#importing librarries
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing

#reading datasets
bank=pd.read_csv("/content/bank.csv",index_col=0,sep=';')
bank = bank.drop('duration',axis=1)

#display raw dataset
bank.head()

#display attributes
print(bank.keys())

bank.columns

#finding data to be encoded 
bank_categorial=bank.loc[:,bank.dtypes==np.object]
bank_categorial.head()

bank_numeric=bank.loc[:,bank.dtypes!=np.object]
bank_numeric.head()

#trying outlier detection on a column
sns.boxplot(x=bank['balance'])

k= np.abs(stats.zscore(bank['balance']))
print(k)

threshold=3
print(np.where(k>3))

a= np.abs(stats.zscore(bank['previous']))
print(a)

threshold=3
print(np.where(a>3))

#handling missing values
sig={'education','job', 'housing', 'loan'}
for var in sig:
  bank[var+'un']=(bank[var]=='unknown').astype(int)

missing_values = bank.isnull().mean()*100

missing_values.sum()

#check duplicated values
bank[bank.duplicated(keep=False)]

#to check numeric values
bank_numeric=bank.loc[:,bank.dtypes!=np.object]
bank_numeric.head()

#data categorial value encoding 
le = preprocessing.LabelEncoder()
bank.job = le.fit_transform(bank.job)
bank.marital = le.fit_transform(bank.marital)
bank.education = le.fit_transform(bank.education)
bank.default = le.fit_transform(bank.default)
bank.housing = le.fit_transform(bank.housing)
bank.loan = le.fit_transform(bank.loan)
bank.contact = le.fit_transform(bank.contact)
bank.month = le.fit_transform(bank.month)
bank.poutcome = le.fit_transform(bank.poutcome)
bank.y = le.fit_transform(bank.y)
bank.head()

#outlier detection using boxplot
sns.boxplot(x=bank)

scaler = MinMaxScaler()
scaler.fit(bank)

#normalize data
normalized_X = preprocessing.normalize(bank)
normalized_X

standardized_X = preprocessing.scale(bank)
standardized_X