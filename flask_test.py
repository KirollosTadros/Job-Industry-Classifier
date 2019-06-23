from flask import Flask, jsonify, request
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, svm
import pickle

app = Flask(__name__)

@app.route('/', methods = ['GET', 'POST'] )
def index():
	if (request_method == 'POST'):
		sone_json =  request.get_json()
		return jsonify({'you sent' : sone_json }), 201
	else:
		return jsonify({"about" : "Hellow World!"})

@app.route('/industry/<string:title>', methods = ['GET'])
def industry(title):

	SVM = pickle.load(open("finalized_model.sav", 'rb'))
	tfidf = pickle.load(open("feature.pkl", 'rb' ) )

	
	x_test = [title]
	Train_X_Tfidf = tfidf.transform(x_test)
	predictions_SVM = SVM.predict(Train_X_Tfidf)
	if(predictions_SVM ==[2]):
		return jsonify ({'result':"IT"})

	if(predictions_SVM ==[3]):
		return jsonify ({'result':"Marketing"})

	if(predictions_SVM ==[1]):
		return jsonify ({'result':"Education"})

	if(predictions_SVM ==[0]):
		return jsonify ({'result':"Accountancy"})
	

if __name__ == '__main__':
	app.run(debug=True)
