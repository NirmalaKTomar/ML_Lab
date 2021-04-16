import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

f=open ('/content/bank-names.txt','r', errors="ignore")
raw=f.read()
print ("\t\t\tititreading bank-names.txt file")
print (raw)

B_full=pd.read_csv("/content/bank-full.csv",index_col=0, sep = ';') 
Bank=pd.read_csv('/content/bank.csv',index_col=0, sep = ';')

B_full.head()
Bank.head()

Bank.head()

y = pd.get_dummies(Bank['y'], columns = ['y'],prefix = ['y'], drop_first= True)
Bank.head()

#printing with 7 records
y=Bank.iloc[ :7]
print("\t\t\tprinting 7 records") 
print(y)

print("\t\t\t\t\tprinting top 5 out of 7 records")
y.head()

print('Jobs: \n', Bank['job'].unique(),'\n')
print('Jobs:\n', Bank['job'].unique(),'\n')
print('education:\n', Bank['education'].unique())

print("\t\tExploring and demonstrating Python")
import pandas as pd
s = pd.Series([1, 3, 5, 10, 6, 8])
print("Pandas dataseries\n", s)

df = pd.DataFrame(np.random.randn(9, 4),columns=['a','b','c','d'])
print("Pandas DataFrame \n", df)

print("To access a specific element using iloc  \n", df.iloc[0,1])

print("Acessing a specific section of data\n", df.iloc[0:3,0:2])

print("Printing the top section of the dataset \n",df.head(5))

print("Printing the bottom section of the dataset \n",df.tail(3))

print("Summary of the dataset \n", df.describe())

print(df.to_numpy())
