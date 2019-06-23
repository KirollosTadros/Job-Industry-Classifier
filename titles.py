
from sklearn.naive_bayes import MultinomialNB
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, svm
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils import class_weight


np.random.seed(500)


#Dataset
jobs_data = pd.read_csv (r"Jobtitles.csv",encoding='latin-1')
class_0_count, class_1_count, class_2_count, class_3_count = jobs_data.industry.value_counts()




#feature to be used
X=jobs_data['job title']

#Classes
y= jobs_data['industry']

#Enocding into numeric value
Encoder = LabelEncoder()
y = Encoder.fit_transform(y)

'''
class_weight = {"IT": 1.,
                "Marketing": 2.,
                "Education": 3.,
                "Accountancy":11.}
'''






#Data Split
Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(X,y)





'''
#Enocding into numeric value
Encoder = LabelEncoder()
Train_Y = Encoder.fit_transform(Train_Y)
Test_Y = Encoder.fit_transform(Test_Y)
'''

#converting data to numeric feature vector
Tfidf_vect = TfidfVectorizer(max_features=5000)
Tfidf_vect.fit(X)
Train_X_Tfidf = Tfidf_vect.transform(Train_X)
Test_X_Tfidf = Tfidf_vect.transform(Test_X)



print(Train_X_Tfidf)

'''
class_weights = class_weight.compute_class_weight('balanced',
                                                 np.unique(Train_Y),
                                                 Train_Y)

print(class_weights)'''


class_weights = {0: 1,
                 1: 2,
                 2: 4,
                 3: 6}


# fit the training dataset on the NB classifier
ctf = MultinomialNB()


ctf.fit(Train_X_Tfidf,Train_Y)
# predict the labels on validation dataset
predictions_NB = ctf.predict(Test_X_Tfidf)


# Use accuracy_score function to get the accuracy
print("Naive Bayes Accuracy Score -> ",accuracy_score(predictions_NB, Test_Y)*100)





print(f1_score(Test_Y, predictions_NB, average='weighted'))


# Classifier - Algorithm - SVM
# fit the training dataset on the classifier
SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto', class_weight = 'balanced')
SVM.fit(Train_X_Tfidf,Train_Y)
# predict the labels on validation dataset
predictions_SVM = SVM.predict(Test_X_Tfidf)
# Use accuracy_score function to get the accuracy
print("SVM Accuracy Score -> ",accuracy_score(predictions_SVM, Test_Y)*100)

print(
pd.crosstab(
    pd.Series(Test_Y, name='Actual'),
    pd.Series(predictions_SVM, name='Predicated'),
    margins=True
)
)

print(f1_score(Test_Y, predictions_SVM, average=None))

