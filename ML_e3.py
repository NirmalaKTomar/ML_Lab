print("exhibit the working of the decision tree based ID3 algorithm,build the decision tree and classify a new sample.")
#importing librarries
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
import math
eps = np.finfo(float).eps
from numpy import log2 as log

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

k=bank.describe()
k

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

bank.isna().any()

#check duplicated values
bank[bank.duplicated(keep=False)]

#to check numeric values
bank_numeric=bank.loc[:,bank.dtypes!=np.object]
bank_numeric.head()

nbank=bank
#data categorial value encoding 
le = preprocessing.LabelEncoder()
nbank.job = le.fit_transform(bank.job)
nbank.marital = le.fit_transform(bank.marital)
nbank.education = le.fit_transform(bank.education)
nbank.default = le.fit_transform(bank.default)
nbank.housing = le.fit_transform(bank.housing)
nbank.loan = le.fit_transform(bank.loan)
nbank.contact = le.fit_transform(bank.contact)
nbank.month = le.fit_transform(bank.month)
nbank.poutcome = le.fit_transform(bank.poutcome)
nbank.y = le.fit_transform(bank.y)
nbank.head()

#outlier detection using boxplot
sns.boxplot(x=nbank)

#scaler = MinMaxScaler()
#scaler.fit(bank)

#normalize data
normalized_X = preprocessing.normalize(nbank)
normalized_X

standardized_X = preprocessing.scale(nbank)
standardized_X

print('check 1: ', bank.groupby(['loan','y']).size())
print('\n\ncheck 2: ', bank.groupby(['job','y']).size())
print('\n\ncheck 3: ', bank.groupby(['marital','y']).size())
print('\n\ncheck 4: ', bank.groupby(['education','y']).size())
print('\n\ncheck 5: ', bank.groupby(['poutcome','y']).size())
print('\n\ncheck 6: ', bank.groupby(['month','y']).size())
print('\n\ncheck 7: ', bank.groupby(['y', 'default']).size())

def entropy(probs):  
 import math
 return sum( [-prob*math.log(prob, 2) for prob in probs] )

#Function to calculate Probabilities of positive and negative examples 

def entropy_of_list(a_list):
 from collections import Counter
 cnt = Counter(x for x in a_list) #Count the positive and negative ex
 num_instances = len(a_list)
#Calculate the probabilities that we required for our entropy formula 
 probs = [x / num_instances for x in cnt.values()] 

#Calling entropy function for final entropy 

 return entropy(probs)
total_entropy = entropy_of_list(bank['job'])

print("\n Total Entropy of bank marketing Data Set:",total_entropy)

entropy_node = 0  #Initialize Entropy
values = bank.loan.unique()  #Unique objects - 'Yes', 'No'
for value in values:
    fraction = bank.loan.value_counts()[value]/len(bank.loan)  
    entropy_node += -fraction*np.log2(fraction)

entropy_node

bank[bank.index.duplicated()]
bank_n = bank[~bank.index.duplicated()]
bank_n

bank[nbank.index.duplicated()]
bank2k = nbank[~nbank.index.duplicated()]
bank2k=bank2k.head()
bank2k

def information_gain(df, split_attribute_name, target_attribute_name, trace=0):
 print("Information Gain Calculation of ",split_attribute_name)
 print("target_attribute_name",target_attribute_name)

#Grouping features of Current Attribute

 df_split = df.groupby(split_attribute_name)
 for name,group in df_split:
         print("Name: ",name)
         print("Group: ",group)
 nobs = len(df.index) * 1.0
 print("NOBS",nobs)
 df_agg_ent = df_split.agg({target_attribute_name : [entropy_of_list, lambda x: len(x)/nobs] })[target_attribute_name]
 print("df_agg_ent",df_agg_ent)
 df_agg_ent = df_split.agg({target_attribute_name : [entropy_of_list, lambda x: len(x)/nobs] })[target_attribute_name]
 print("df_agg_ent",df_agg_ent)

# Calculate Information Gain
 avg_info = sum( df_agg_ent[entropy] * df_agg_ent['Prob1'] )
 old_entropy = entropy_of_list(df[target_attribute_name])
 return old_entropy - avg_info

print('Info-gain for Outlook is :'+ str(information_gain(bank_n, 'job', 'jobun')),"\n")

def find_entropy(df):
    Class = df.keys()[-1]   #To make the code generic, changing target variable class name
    entropy = 0
    values = df[Class].unique()
    for value in values:
        fraction = df[Class].value_counts()[value]/len(df[Class])
        entropy += -fraction*np.log2(fraction)
    return entropy
  
  
def find_entropy_attribute(df,attribute):
  Class = df.keys()[-1]   #To make the code generic, changing target variable class name
  target_variables = df[Class].unique()  #This gives all 'Yes' and 'No'
  variables = df[attribute].unique()    #This gives different features in that attribute (like 'Hot','Cold' in Temperature)
  entropy2 = 0
  for variable in variables:
      entropy = 0
      for target_variable in target_variables:
          num = len(df[attribute][df[attribute]==variable][df[Class] ==target_variable])
          den = len(df[attribute][df[attribute]==variable])
          fraction = num/(den+eps)
          entropy += -fraction*log(fraction+eps)
      fraction2 = den/len(df)
      entropy2 += -fraction2*entropy
  return abs(entropy2)


def find_winner(df):
    Entropy_att = []
    IG = []
    for key in df.keys()[:-1]:
        IG.append(find_entropy(df)-find_entropy_attribute(df,key))
    return df.keys()[:-1][np.argmax(IG)]
  
  
def get_subtable(df, node,value):
  return df[df[node] == value].reset_index(drop=True)


def buildTree(df,tree=None): 
    Class = df.keys()[-1]
    node = find_winner(df)
    attValue = np.unique(df[node])  
    if tree is None:                    
        tree={}
        tree[node] = {}
    for value in attValue:
        
        subtable = get_subtable(df,node,value)
        clValue,counts = np.unique(subtable[Class],return_counts=True)                        
        
        if len(counts)==1:
            tree[node][value] = clValue[0]                                                    
        else:        
            tree[node][value] = buildTree(subtable) 
    return tree

t=buildTree(bank2k)
import pprint
pprint.pprint(t)