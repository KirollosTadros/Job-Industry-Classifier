import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, svm

import pickle
SVM = pickle.load(open("finalized_model.sav", 'rb'))
tfidf = pickle.load(open("feature.pkl", 'rb' ) )
#tf_new = TfidfVectorizer(max_features = 500000, vocabulary = tfidf.vocabulary_)

x_test="nodejs"
x_test = [x_test]
Train_X_Tfidf = tfidf.transform(x_test)
predictions_SVM = SVM.predict(Train_X_Tfidf)
if(predictions_SVM ==[2]):
	print ("IT")
print(predictions_SVM)
