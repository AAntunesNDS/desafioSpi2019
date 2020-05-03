# -*- coding: utf-8 -*-
from flask import Flask, render_template, flash, request, Markup
from flask_wtf import FlaskForm
from wtforms import DecimalField, IntegerField, validators, SubmitField

import regression_models


# App config.
DEBUG = True
app = Flask(__name__)
app.config.from_object(__name__)
app.config['SECRET_KEY'] = '7d441f27d441f27567d441f2b6176a'


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
	
	columns_name = regression_models.columns_data[:-1]

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
		arr = [[crim, zn, indus, chas, nox, rm, age, dis, rad, tax, ptratio, b, lstat]]
		predict_value = regression_models.predict_price(arr)

		if form.validate():
			# Save the comment here.
			flash('{}'.format(predict_value))
		else:
			flash(form.errors)

	return render_template('hello.html', form=form, hd=columns_name)

@app.route('/grafico')
def bar():
	response = regression_models.compute_rmse_regressors()

	for regressor_name, rmse in response.items():
		labels = regressor_name
		bar_values = rmse

	return render_template('grafico.html', title='Grafico de classificadores',
							max=20, labels=labels, values=bar_values)


if __name__ == "__main__":
	app.run()
