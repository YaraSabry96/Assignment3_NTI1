# coding: utf-8
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import r2_score,mean_squared_error


class Model1:
	"""docstring for Moddel"""
	def __init__(self):
		#hna ha7ot elobjects elli ha5odha
		self.sc = StandardScaler()
		self.sex_encoder = LabelEncoder()
		self.schoolsup_encoder = LabelEncoder()
	def read_df(self,path):
		self.dataset = pd.read_csv(path)
		#self 3shan el dataset tp2a mtshafa fe kol class 

	def label_encoding(self):
		self.dataset =self.dataset[['sex','age','studytime','failures','schoolsup']]
		self.dataset['sex'] = self.sex_encoder.fit_transform(self.dataset['sex'].values)
		self.dataset['schoolsup'] = self.schoolsup_encoder.fit_transform(self.dataset['schoolsup'].values)
		

	def split_df(self):
		self.x = self.dataset.iloc[:, 0:4].values
		self.y = self.dataset.iloc[:, 4].values

	def scaling(self):
		self.x = self.sc.fit_transform(self.x)

	def train_test(self,test_size):
		self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x, self.y, test_size = test_size, random_state = 0)

	def train(self,modle_name):
		self.read_df("C:\\Users\\Yara Sabry\\Desktop\App\\students.csv")
		self.label_encoding()
		self.split_df()
		self.scaling()
		self.train_test(0.25)
		if modle_name == 'Logistic':
			self.classifier = LogisticRegression()
			self.classifier.fit(self.x_train, self.y_train)	
		elif modle_name == 'KNN':
			self.classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
			self.classifier.fit(self.x_train, self.y_train)	
		elif  modle_name == 'SVM':
			self.classifier = SVC(kernel = 'linear', random_state = 0)
			self.classifier.fit(self.x_train, self.y_train)	
		elif modle_name == 'NB':
			self.classifier = GaussianNB()
			self.classifier.fit(self.x_train, self.y_train)	
		elif modle_name == 'DT':
			self.classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
			self.classifier.fit(self.x_train, self.y_train)
		elif modle_name == 'RF':
			self.classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
			self.classifier.fit(self.x_train, self.y_train)
		

	def evaluate(self):
		y_pred = self.classifier.predict(self.x_test)
		#r2 = r2_score(self.y_test,y_pred)
		acc = self.classifier.score(self.x_test,self.y_test)
		#mse = mean_squared_error(y_pred,self.y_test)
		return acc

	def seralize(self):
		save_pickle = open("model.pickle","wb")
		pickle.dump(self.classifier,save_pickle)
		save_pickle.close()

	def predict(self,test):
		test = self.sc.transform([test])
		return self.classifier.predict(test)

if __name__ == '__main__':
	yar = Model1()
	yar.train('KNN')
	print(yar.evaluate())
	yar.seralize()
	print(yar.predict([0,17,2,0]))