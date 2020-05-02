# -*- coding: utf-8 -*-
from flask import Flask, render_template, flash, request, Markup
from flask_wtf import FlaskForm
from wtforms import DecimalField, IntegerField, validators, SubmitField

import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn import svm
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.model_selection import train_test_split


# App config.
DEBUG = True
app = Flask(__name__)
app.config.from_object(__name__)
app.config['SECRET_KEY'] = '7d441f27d441f27567d441f2b6176a'


features = ['CRIM','ZN','INDUS',
			'CHAS','NOX','RM',
			'AGE','DIS','RAD',
			'TAX','PTRATIO','B',
			'LSTAT']


class ReusableForm(FlaskForm):
	crim = DecimalField('CRIM')
	zn = DecimalField('ZN')
	indus = DecimalField('INDUS')
	chas = IntegerField('CHAS')
	nox = DecimalField('NOX')
	rm = DecimalField('RM')
	age = DecimalField('AGE')
	dis = DecimalField('DIS')
	rad = IntegerField('RAD')
	tax = DecimalField('TAX')
	ptratio = DecimalField('PTRATIO')
	b = DecimalField('B')
	lstat = DecimalField('LSTAT')

@app.route("/", methods=['GET', 'POST'])
def hello():
	form = ReusableForm(request.form)
	if form.validate_on_submit():
		crim=float(request.form['CRIM'])
		zn=float(request.form['ZN'])
		indus=float(request.form['INDUS'])
		chas=int(request.form['CHAS'])
		nox=float(request.form['NOX'])
		rm=float(request.form['RM'])
		age=float(request.form['AGE'])
		dis=float(request.form['DIS'])
		rad=int(request.form['RAD'])
		tax=float(request.form['TAX'])
		ptratio=float(request.form['PTRATIO'])
		b=float(request.form['B'])
		lstat=float(request.form['LSTAT'])

		#array to predict clf
		arr = np.array([[crim, zn, indus, chas, nox, rm, age, dis, rad, tax, ptratio, b, lstat]])
		
		if form.validate():
		# Save the comment here.
			flash('{}'.format(clf.predict(arr)))
		else:
			flash(form.errors)

	return render_template('hello.html', form=form, hd=features)


def compute_clf(x_train, x_test, y_train, y_test):
	classifiers = [
	    svm.SVR(),
	    linear_model.BayesianRidge(),
	    linear_model.LassoLars(),
	    linear_model.ARDRegression(),
	    linear_model.PassiveAggressiveRegressor(),
	    linear_model.TheilSenRegressor(),
	    linear_model.LinearRegression()
		]

	values = []

	for clf in classifiers:
	    clf.fit(x_train, y_train)
	    y_pred = clf.predict(x_test)
	    values.append((sqrt(mean_squared_error(y_pred, y_test))))

	return values



@app.route('/grafico')
def bar():
	labels = ['SVM.Svr', 'Bayesian Rigde',
		'LassoLars', 'ARDRegression', 'PassiveAgressiveRegressor',
		'TheilSenRegressor', 'LinearRegression']

	bar_values = values
	return render_template('grafico.html', title='Grafico de classificadores',
							max=20, labels=labels, values=bar_values)


if __name__ == "__main__":
	hd = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT','MEDV']
	data = pd.read_csv('housing.data', header=None, delim_whitespace=True)
	data.columns = hd

	# label
	y = data['MEDV']

	# features
	X = data.loc[:, data.columns != 'MEDV']

	# Regression config.
	clf = linear_model.LinearRegression()


	clf.fit(X, y)

	x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=13)
	values = compute_clf(x_train, x_test, y_train, y_test)

	app.run()
