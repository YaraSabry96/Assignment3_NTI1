from flask import Flask,render_template
from flask import jsonify,request
from model import Model1


# LR name of file , Model name of class
clf1 = Model1()

# 5at object mn el class elli hya Model fe clf
app = Flask(__name__)
#@app.route("/home")
#def index():
#    return render_template('from.html')

@app.route("/read")# app dh name of file bta3ii ,route dh hy3ml path 
def read():
	clf1.read_df("C:\\Users\\Yara Sabry\\Desktop\App\\students.csv")
	return clf1.dataset.head().to_json()

@app.route("/label")
def labelencoding():
	clf1.label_encoding()
	return "Data label Done!"

@app.route("/split")
def split():
	clf1.split_df()
	return "Data split Done!"

@app.route("/scale")
def scale():
	clf1.scaling()
	return "scaling Done!"

@app.route("/train_test")
def train_test():
	clf1.train_test(0.25)
	
	return "train_test Done!"

@app.route("/train",methods=["GET"])
def train():
	#name = request.args.get('name')
	#logistic = request.args.get('Logistic')
	#knn = request.args.get('KNN')
	#svm = request.args.get('SVM')
	#nb = request.args.get('NB')
	#dt = request.args.get('DT')
	#rf = request.args.get('RF')
	#if name == logistic :
	#	clf1.train()
		#print("Training logistic done!")
		#print("logistic done!")
	#elif  name == knn:
	#	clf2.train()
	#elif name == svm:
	#	clf3.train()
	#elif name == nb:
	#	clf4.train()	
	#elif name == dt:
	#	clf5.train()
	#elif name == rf:
	#	clf6.train()
	model_name = request.args.get('name')
	clf1.train(model_name)
	#return "Training done!"
	acc = clf1.evaluate()
	return "<h1> the accuracy Using "+model_name+" Model :\
	 Accuracy:<br>\
  <h3>"+str(acc)+"</h3>\
  <br>\
  R2:<br>\
  <h3>"+str(1)+"</h3>\
  <br>\
    MSE:<br>\
  <h3>"+str(1)+"</h3>\
  <form action='http://127.0.0.1:9090/test'>\
    <input type='submit' value='Go to accuracy'>\
    </form>"

@app.route("/test")
def test():
	return "<form action='http://127.0.0.1:9090/predict'>\
   <h2> Sex </h2>\
  <input type='text' name='sex'>\
   <h2> Age </h2>\
  <input type='text' name='age'>\
   <h2> StudyTime </h2>\
  <input type='text' name='studytime'>\
   <h2> Failures </h2>\
  <input type='text' name='failures'>\
  <input type='submit' value = 'predict'>\
  </form>"

@app.route("/predict",methods=["GET"])
def predict():
	sex = int(request.args.get('sex'))
	age = int(request.args.get('age'))
	studytime = int(request.args.get('studytime'))
	failures = int(request.args.get('failures'))
	y_pred = clf1.predict([sex,age,studytime,failures])
	resp = {"class":int(y_pred[0])}
	#text = "<h2> The Score :"+ resp+"</h2>"
	#return text
	return jsonify(resp)




if __name__ == '__main__':
	try:
		app.run(port=9090,host='127.0.0.1')
	except Exception as e:
		print(e)


