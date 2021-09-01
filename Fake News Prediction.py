#!/usr/bin/env python
# coding: utf-8

# ## Importing Packages and NLP related data.

# In[1]:


import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer


# * Regular Expression: re is for searching text in documents
# * nltk means Natural Language Toolkit.
# * PorterSteemer is used in steeming, stemming is a process of getting the root word from a normal word, ex: root word for learning. learner, prelearning is learn, i.e. stemming means removinmg all possible prefixes and suffixes of a word.
# * TfidfVectorizer wil used to form feature vectors from texts.
# * Stopwords are the words that dont add any information to the sentense like the, on, etc.

# In[2]:


import nltk
nltk.download('stopwords')


# ## Data Preprocessing

# In[3]:


data_set = pd.read_csv('/media/shreyashkharat/Storage Drive/Machine Learning, Deep Learning/Python/Projects/Fake News Predicition/resources/train.csv', header = 0)


# In[4]:


data_set.shape


# In[5]:


data_set.info()


# * The above infos clearly show that there are missing values in title, author, text.

# ### Missing Value Imputation

# In[6]:


data_set = data_set.fillna(' ')


# In[7]:


data_set['describe'] = data_set['title'] + ' ' + data_set['author']


# In[8]:


x = data_set.loc[:, data_set.columns != 'label']


# In[9]:


y = data_set['label']


# ## Stemming

# In[10]:


port_stem = PorterStemmer()


# In[11]:


def stemming(argument):
    stemmed_argument = re.sub('[^a-zA-Z]', ' ', argument)
    stemmed_argument = stemmed_argument.lower()
    stemmed_argument = stemmed_argument.split()
    stemmed_argument = [port_stem.stem(word) for word in stemmed_argument if not word in stopwords.words('english')]
    stemmed_argument = ' '.join(stemmed_argument)
    return stemmed_argument


# Explanation of above function:
# * Line 1: It differentiates between alphabets and all other characters, i.e. it considers only characters a-z and A-Z from describe, and replace any other character by a space.
# * Line 2: It converts all letters into lower case.
# * Line 3: It splits the letters in respective lists.
# * Line 4: Now we stem each word all the non-stopwords.
# * Line 5: We join all the words using space.
# * Line 6: Return result.

# In[12]:


data_set['describe'] = data_set['describe'].apply(stemming)


# ## Variable Extraction and Data Vectorization.

# In[13]:


x = data_set['describe'].values
y = data_set['label'].values


# * Data Vectorization

# In[14]:


vectorizer = TfidfVectorizer()


# In[15]:


vectorizer.fit(x)
x = vectorizer.transform(x)


# ### Train-Test Split

# In[16]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)


# ## Training the Model and Evaluating it.

# In[17]:


from sklearn.linear_model import LogisticRegression
model_logi = LogisticRegression()


# In[18]:


model_logi.fit(x_train, y_train)


# In[19]:


train_pred_logi = model_logi.predict(x_train)
from sklearn.metrics import accuracy_score
accuracy_score(y_train, train_pred_logi)


# In[20]:


test_pred_logi = model_logi.predict(x_test)
accuracy_score(y_test, test_pred_logi)


# In[21]:


from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,test_pred_logi)


# * The model seems pretty good, with a excellent accuracy of 0.9784 on test_set.

# In[ ]:




