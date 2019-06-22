#!/usr/bin/env python
# coding: utf-8

# In[2]:


from imblearn.over_sampling import SMOTE
from sklearn.naive_bayes import MultinomialNB
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, svm
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils import class_weight


# ### 1.0 Loading our dataset

# In[3]:


jobs_data = pd.read_csv (r"Jobtitles.csv",encoding='latin-1')
class_0_count, class_1_count, class_2_count, class_3_count = jobs_data.industry.value_counts()


# In[4]:


#feature to be used
X=jobs_data['job title']

#Classes
y= jobs_data['industry']


# ## 1.1 Encoding the Classes
# Encdoing classes to convert classes into numerical values

# In[5]:


#Enocding into numeric value
Encoder = LabelEncoder()
y = Encoder.fit_transform(y)


# ## 1.2 Split Data
# Split Data into train and test set

# In[6]:


Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(X,y)


# ## 1.3 Data Vectorization

# In[7]:


#converting data to numeric feature vector
Tfidf_vect = TfidfVectorizer(max_features=5000)
Tfidf_vect.fit(X)
Train_X_Tfidf = Tfidf_vect.transform(Train_X)
Test_X_Tfidf = Tfidf_vect.transform(Test_X)


# ### 2.0 classifiers
# In this part we are going to try different classifiers to decide which one is the best
# 

# ## 2.1 Multinomial Naive Bayes
# We will start with Multinomial Naive Bayes

# In[8]:


NB = MultinomialNB()
NB.fit(Train_X_Tfidf,Train_Y)
predictions_NB = NB.predict(Test_X_Tfidf)


# ### 2.1.1 Crosstab

# We will use Crosstab to see how our predication is close to the correct one

# In[9]:


print(
pd.crosstab(
    pd.Series(Test_Y, name='Actual'),
    pd.Series(predictions_NB, name='Predicated'),
    margins=True
))


# ### 2.1.2 F1 Score
# We will use F1 Score to know how close we are to the correct classification 

# In[10]:


print("F1 Score: {0:.2f}".format(f1_score(Test_Y, predictions_NB, average='weighted')))


# ## 2.2 SVM Classifier
# Now we will try SVM Classifier

# In[11]:


SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
SVM.fit(Train_X_Tfidf,Train_Y)
predictions_SVM = SVM.predict(Test_X_Tfidf)


# ### 2.2.1 SVM Crosstab

# In[12]:


print(
pd.crosstab(
    pd.Series(Test_Y, name='Actual'),
    pd.Series(predictions_SVM, name='Predicated'),
    margins=True
))


# ### 2.2.2 SVM F1 Score

# In[30]:


print("F1 Score: {0:.4f}".format(f1_score(Test_Y, predictions_SVM, average='weighted')))


# # 3.0 Balancing data
# Now we will deal with imbalanced data to improve accuarcy 

# ## 3.1 Class Weight
# We will use class Weight to give more weight to the low frequency classes

# In[59]:


class_weights = {0: 4,
                 1: 3,
                 2: .8,
                 3: 1.25}


# ## 3.2 SVM Classifier with weighted Classes

# In[60]:


SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto', class_weight = class_weights)
SVM.fit(Train_X_Tfidf,Train_Y)
predictions_SVM = SVM.predict(Test_X_Tfidf)


# ## 3.3 Crosstab
# Crosstab of the new SVM classifier

# In[61]:


print(
pd.crosstab(
    pd.Series(Test_Y, name='Actual'),
    pd.Series(predictions_SVM, name='Predicated'),
    margins=True
))


# In[62]:


print("F1 Score: {0:.4f}".format(f1_score(Test_Y, predictions_SVM, average='weighted')))

