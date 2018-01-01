
# coding: utf-8

# In[1]:

import numpy as np
from nltk.stem.porter import *
import sys


# In[2]:

##dictionary##
arg1 = sys.argv[1]
arg2 = sys.argv[2]
data=open('ass1-train_data.txt','r')
whitelist = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0 ')
all_lines=data.readlines()
stemmer = PorterStemmer()
dictionary={}
count=0
for line in all_lines:
    x=line.split('\t')
    #check for valid string
    if((x[0]=='0' or x[0]=='1') and any(c.isalpha() for c in x[1])):
        x=''.join(filter(whitelist.__contains__, x[1])).split()
        for words in x:
            if(len(words)>2):
                words=words.lower()
                st_word=stemmer.stem(words)
                dictionary[st_word]=count
                count=count+1         
##dictionary


# In[3]:

##feature vector (fv) creation##
data2=open('ass1-train_data.txt','r')
##  test data -> matrix  ##
all_lines=data2.readlines()
bag=[]
label=0
v=0
for line in all_lines:
    x=line.split('\t')
    if((x[0]=='0' or x[0]=='1') and any(c.isalpha() for c in x[1])):
        v=v+1
        ##print(v)
        y=np.zeros(count)
        ##print(type(x[0]))
        label=np.append(label,int(x[0]))
        x=''.join(filter(whitelist.__contains__, x[1])).split()
        ##print(x)
        for words in x:  
            words=words.lower()
            st_word=stemmer.stem(words)
            if st_word in dictionary and len(words)>2:
                k=dictionary[st_word]
                y[k]=y[k]+1
        bag.append(y)
training_fv=np.asarray(bag)
training_lab=np.asarray(label[0:label.size-1])


# In[4]:

##test data##
data3=open(arg1,'r')
all_lines=data3.readlines()
test_bag=[]
test_label=0
for line in all_lines:
    x=line
    if(any(c.isalpha() for c in x)):
        y=np.zeros(count)
        x=''.join(filter(whitelist.__contains__, x)).split()
        ##print(x)
        for words in x:
            words=words.lower()
            st_word=stemmer.stem(words)
            if st_word in dictionary and len(words)>2:
                k=dictionary[st_word]
                y[k]=y[k]+1                   
        test_bag.append(y)
test_fv=np.asarray(test_bag)


# In[5]:



# In[6]:

##fit the clasifier ##
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()
clf.fit(training_fv, training_lab)
print("Naive Bayes classifier for multinomial model.")
print("Training data accuracy")
print(clf.score(training_fv,training_lab))
thelist = clf.predict(test_fv)
thefile=open(arg2,'w')
for item in thelist:
  thefile.write("%s\n" % item)
print("Output file created.\n")

# In[ ]:



