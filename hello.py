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

import json
import plotly
import plotly.offline as py
import plotly.graph_objs as go


# App config.
DEBUG = True
app = Flask(__name__)
app.config.from_object(__name__)
app.config['SECRET_KEY'] = '7d441f27d441f27567d441f2b6176a'

# Regression config.
clf = linear_model.LinearRegression()

labels = ['SVM.Svr', 'Bayesian Rigde', 
		'LassoLars', 'ARDRegression', 'PassiveAgressiveRegressor',
		'TheilSenRegressor', 'LinearRegression']

values = []


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
		crim=float(request.form['crim'])
		zn=float(request.form['zn'])
		indus=float(request.form['indus'])
		chas=int(request.form['chas'])
		nox=float(request.form['nox'])
		rm=float(request.form['rm'])
		age=float(request.form['age'])
		dis=float(request.form['dis'])
		rad=int(request.form['rad'])
		tax=float(request.form['tax'])
		ptratio=float(request.form['ptratio'])
		b=float(request.form['b'])
		lstat=float(request.form['lstat'])
		
		#array to predict clf
		arr = np.array([[crim, zn, indus, chas, nox, rm, age, dis, rad, tax, ptratio, b, lstat]])

		if form.validate():
		# Save the comment here.
			flash('{}'.format(clf.predict(arr)))
		else:
			flash(form.errors)

	return render_template('hello.html', form=form)


def compute_clf(x_train, x_test, y_train, y_test):
	classifiers = [
	    svm.SVR(),
	    linear_model.BayesianRidge(),
	    linear_model.LassoLars(),
	    linear_model.ARDRegression(),
	    linear_model.PassiveAggressiveRegressor(),
	    linear_model.TheilSenRegressor(),
	    linear_model.LinearRegression()]

	values = []

	for clf in classifiers:
	    clf.fit(x_train, y_train)
	    y_pred = clf.predict(x_test)
	    values.append((sqrt(mean_squared_error(y_pred, y_test))))

	return values



@app.route('/grafico')
def bar():
	bar_labels=labels
	bar_values=values
	return render_template('grafico.html', title='Grafico de classificadores', max=20, labels=bar_labels, values=bar_values)


if __name__ == "__main__":
	hd = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT','MEDV']
	data = pd.read_csv('housing.data', header=None, delim_whitespace=True)
	data.columns = hd

	# label
	y = data['MEDV']
	
	# features
	X = data.loc[:, data.columns != 'MEDV']

	clf.fit(X, y)

	x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=13)
	values = compute_clf(x_train, x_test, y_train, y_test)

	app.run()