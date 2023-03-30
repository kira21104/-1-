import flask
import numpy
from flask import render_template
import keras
import pickle
import sklearn
import joblib
from pyngrok import ngrok
from flask import Flask
import flask
from joblib import load
from sklearn.linear_model import LinearRegression

app = flask.Flask(__name__, template_folder = 'templates')

@app.route('/', methods = ['POST', 'GET'])

@app.route('/index', methods = ['POST', 'GET'])
def main():
    if flask.request.method == 'GET':
        return render_template('main.html')
    
    if flask.request.method == 'POST':

        with open('encoder.joblin', 'rb') as f:
            encoder_model = joblib.load(f)

        with open('minmax_x.joblin', 'rb') as f:
            mimax_x_model = joblib.load(f)

        loaded_model = keras.models.load_model('../models4')

        with open('minmax_y.joblin', 'rb') as f:
            minmax_y_model = joblib.load(f)

        plotnost = float(flask.request.form['density'])
        mod_upr = float(flask.request.form['modulus_of_elasticity'])
        kol_otv = float(flask.request.form['amount_of_hardener'])
        epoxy_g = float(flask.request.form['content_of_epoxy_groups'])
        temp = float(flask.request.form['flash_point'])
        pov_plotnost = float(flask.request.form['surface_density'])
        upr_rast = float(flask.request.form['tensile_modulus_of_elasticity'])
        proch_rast = float(flask.request.form['tensile_strength'])
        smola = float(flask.request.form['resin_consumption'])
        ugol = int(flask.request.form['stripe_angle'])
        shag = float(flask.request.form['stripe_step'])
        plot_nash = float(flask.request.form['stripe_density'])

        #Закодируем угол нашивки
        ugol_enc = encoder_model.transform([0])
        #Добавим все значения в массив
        priznaki = [plotnost, mod_upr, kol_otv, epoxy_g, temp, pov_plotnost, upr_rast, proch_rast, smola, ugol_enc[0], shag, plot_nash]
        #Проведем скалирование признаков
        scal = mimax_x_model.transform(numpy.array(priznaki).reshape(1, -1))
        #Делаем предсказание в нейронной сети
        y_pred = loaded_model.predict(scal).flatten()
        #Дескалируем полученное предсказание
        y = minmax_y_model.inverse_transform([y_pred])
        

        return render_template('main.html', result = y[0][0])

if __name__ == '__main__':
    app.run()
