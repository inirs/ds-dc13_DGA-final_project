
# coding: utf-8

# # DGA DETECTION MODEL

# In[1]:

# IMPORT MODULES

import pandas as pd
import numpy as np
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.cross_validation import cross_val_score


# # Data Ingestion

# In[2]:

#Read Clean Domains
data_clean = pd.read_csv('Alexa_CleanDomains.csv',header=None)
data_clean['domains']=data_clean[0]


# In[3]:

# Strip out xn--

#data_clean['domains'] = data_clean['domains'].map(lambda x: str(x).lstrip('xn--'))


# In[4]:

# READ DGA Domains
data_dga = pd.read_csv('DGADomains.csv',header=None)
data_dga['domains']=data_dga[0]


#  # Data Prep

# In[5]:

# Clean Data without Index
train_clean = pd.DataFrame(data_clean['domains'])
print(train_clean[:10])

len(train_clean)


# In[6]:

#Delete Duplicates

new=train_clean.drop_duplicates()
len(new)
new = new.dropna()


# In[7]:

clean_dom = new[[0]] # 0 are clean

clean_domains1 = clean_dom.values
clean_domains=clean_domains1.tolist()
clean_domains_flat = [val for sublist in clean_domains for val in sublist]


# In[8]:

clean_domains1 = clean_dom.values
clean_domains=clean_domains1.tolist()
clean_domains_flat = [val for sublist in clean_domains for val in sublist]


# In[9]:

count=0
for ll in clean_domains_flat:
    if(len(ll) == 0):
        count+=1
print(count)


# In[10]:

# Clean Data without Index

df_dga = pd.DataFrame(data_dga['domains'])

train_dga=df_dga.drop_duplicates()

dga_dom = train_dga[[0]]
dga_domains1 = dga_dom.values
dga_domains=dga_domains1.tolist()
dga_domains_flat = [val for sublist in dga_domains for val in sublist]


# # FEATURE EXTRACTION

# # FEATURE 1 : Domain Entropy 

# In[11]:

# Shannon entropy H of a given input string.
# Given the discreet random variable X that is a string of N "symbols" consisting of n different characters (n=2 for binary), 
# the Shannon entropy of X in bits/symbol is : where count is the count of character n.

import math
from collections import Counter

def entropy(s):
    p, lns = Counter(s), float(len(s))
    return -sum( count/lns * math.log(count/lns, 2) for count in p.values())


# In[12]:

# Entropy of clean dataset
entropy_clean = []

for l in clean_domains_flat:
    entropy_clean.append(entropy(str(l)))


# In[13]:

# Entropy of dga dataset
entropy_dga = []
for ll in dga_domains_flat:
    entropy_dga.append(entropy(ll))


# In[14]:

print(np.mean(entropy_clean))
print(np.mean(entropy_dga))


# In[15]:

clean_dom['label'] = "0"
dga_dom['label'] = "1"

# Add entropy calculations to dataframes
clean_dom['entropy'] = entropy_clean
dga_dom['entropy'] = entropy_dga


# # FEATURE 2 : Domain Length

# In[16]:

# lexical study
def dom_length(s):
    
    return len(s)


# In[17]:

len_clean = []
for ll in clean_domains_flat:
    len_clean.append(dom_length(str(ll)))


# In[18]:

len_dga = []
for ll in dga_domains_flat:
    len_dga.append(dom_length(ll))


# In[19]:

# Length
clean_dom['length'] = len_clean
dga_dom['length'] = len_dga


# In[20]:

print(np.mean(len_clean))
print(np.mean(len_dga))


# In[ ]:




# # FEATURE 3 : Vowel ratio

# In[21]:

vowels = list("aeiou")
consonants = list("bcdfghjklmnpqrstvxzy")


# In[22]:

# Feature Consonant/length
def conlen_ratio(s):
    number_of_vowels = sum(str(s).count(c) for c in vowels)
    ratio= (number_of_vowels)/(len(str(s))+1)
    return ratio


# In[23]:

# Clean data
cl_ratio_clean = []
for ll in clean_domains_flat:
    cl_ratio_clean.append(conlen_ratio(str(ll)))


# In[24]:

# Clean data
cl_ratio_dga = []
for ll in dga_domains_flat:
    cl_ratio_dga.append(conlen_ratio(ll))
    


# In[25]:

clean_dom.head()


# In[26]:

print(np.mean(cl_ratio_clean))
print(np.mean(cl_ratio_dga))


# In[27]:

# Add ratio calculations to dataframes
clean_dom['ratio'] = cl_ratio_clean
dga_dom['ratio'] = cl_ratio_dga


# # FEATURE 4 : Distinct Characters/length

# In[28]:

0# Feature 4 : Distinct Character
def dom_distchar(s):
 dist_char = ''.join(set(s)) 
 return len(dist_char)/(len(s)+1)


# In[29]:

# Clean
dist_clean = []
for ll in clean_domains_flat:
    dist_clean.append(dom_distchar(str(ll)))
    
dist_dirty = []
for ll in dga_domains_flat:
    dist_dirty.append(dom_distchar(str(ll)))    


# In[30]:

print(np.mean(dist_clean))
print(np.mean(dist_dirty))


# In[31]:

# Add ratio calculations to dataframes
clean_dom['diff_char'] = dist_clean
dga_dom['diff_char'] = dist_dirty
print(len(clean_dom))


# In[32]:

feature_dataset =  pd.concat([clean_dom, dga_dom], ignore_index=True)


# # FEATURE 5: PercentageCoverage
# 

# In[33]:

import nltk
#nltk.download()
english_vocab = set(w.lower() for w in nltk.corpus.words.words())


# In[34]:

def getPercentageCoverage(s):
    count = 0
    tcount = 0
    for i in range(0,(len(s)+1)):
        for j in range(i,(len(s)+1)):
            if s[i:j+1] in english_vocab:
                tcount += 1
            count += 1
    return tcount / float(count) 


# In[35]:

feature_dataset["pctCoverage"] = feature_dataset["domains"].apply(lambda x: getPercentageCoverage(str(x)))


# In[36]:

# FEATURE: NUMBER OF DIGITS PER UNIT LENGTH IN EACH DOMAIN
def num_digits(s):
   o = sum(c.isdigit() for c in s)
   return o

feature_dataset["digits"] = feature_dataset["domains"].apply(lambda x: num_digits(str(x))/(len(str(x))+1))


# In[37]:

char_set = "abcdefghijklmnopqrstuvwxyz-"
for ch in char_set:
    feature_dataset[ch] = feature_dataset["domains"].apply(lambda x: float(str(x).count(ch))/(len(str(x))+1))


# In[38]:

print(feature_dataset[:4])


# # DUMP DGA MODEL FEATURE DATA AS A .CSV FILE

# In[39]:

# Output 
#with open('C:\\Users\\xoc201\\Documents\\Correlation\\feature_dataset.csv','a') as dd:
    #feature_dataset.to_csv(dd, index=False)


# # Random Forest Model

# In[40]:

msk = np.random.rand(len(feature_dataset)) < 0.70

train = feature_dataset[msk]

test = feature_dataset[~msk]

from sklearn.ensemble import RandomForestClassifier
features = feature_dataset.columns[2:45]


# In[41]:

clf = RandomForestClassifier(n_jobs=30, min_samples_leaf = 2, max_features = 17,max_depth=30)
y, _ = pd.factorize(train['label'])


# In[42]:

clf.fit(train[features], y)
predicted = clf.predict(test[features])
print(features)


# In[43]:

y_test, _ = pd.factorize(test['label'])
print(metrics.confusion_matrix(y_test, predicted))


# In[44]:

print(metrics.classification_report(y_test, predicted))


# In[45]:

print(metrics.accuracy_score(y_test, predicted))


# # Feature Importance

# In[ ]:

clf.feature_importances_

# Feature List
# 'entropy', 'length', 'ratio', 'diff_char', 'pctCoverage', 'digits','a', 'b', 'c',
#       'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q',
#       'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '-'


# In[ ]:

import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt


objects = clf.feature_importances_
y_pos = np.arange(len(objects))
 
b1 = plt.bar(y_pos, objects, align='center', alpha=0.5)
#plt.xticks(y_pos, objects)

plt.ylabel('Score')
plt.title('Feature Importance')

plt.show()

# Feature List from feature 0 (=Entropy) to feature 31 
# 'entropy', 'length', 'ratio', 'diff_char', 'pctCoverage', 'a', 'b', 'c',
#       'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q',
#       'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'digits'

