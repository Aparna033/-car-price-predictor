from flask import Flask, render_template, request, redirect
from flask_cors import CORS, cross_origin
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)
cors = CORS(app)

model = pickle.load(open('LinearRegression.pkl', 'rb'))
car = pd.read_csv("CarPrice_Assignment (1).csv")


@app.route('/', methods=['GET','POST'])
def index():

    # Dropdown values
    CarName = sorted(car['CarName'].unique())
    fueltype = sorted(car['fueltype'].unique())
    aspiration = sorted(car['aspiration'].unique())
    doornumber = sorted(car['doornumber'].unique())
    carbody = sorted(car['carbody'].unique())
    drivewheel = sorted(car['drivewheel'].unique())
    enginelocation = sorted(car['enginelocation'].unique())
    enginetype = sorted(car['enginetype'].unique())
    cylindernumber = sorted(car['cylindernumber'].unique())
    fuelsystem = sorted(car['fuelsystem'].unique())

    CarName.insert(0,'Select Car Name')

    return render_template(
        'index.html',

        CarName=CarName,
        fueltype=fueltype,
        aspiration=aspiration,
        doornumber=doornumber,
        carbody=carbody,
        drivewheel=drivewheel,
        enginelocation=enginelocation,
        enginetype=enginetype,
        cylindernumber=cylindernumber,
        fuelsystem=fuelsystem
    )


@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():

     # Numeric Inputs
    car_ID = request.form.get('car_ID')
    symboling = request.form.get('symboling')
    wheelbase = request.form.get('wheelbase')
    carlength = request.form.get('carlength')
    carwidth = request.form.get('carwidth')
    carheight = request.form.get('carheight')
    curbweight = request.form.get('curbweight')
    enginesize = request.form.get('enginesize')
    boreratio = request.form.get('boreratio')
    stroke = request.form.get('stroke')
    compressionratio = request.form.get('compressionratio')
    horsepower = request.form.get('horsepower')
    peakrpm = request.form.get('peakrpm')
    citympg = request.form.get('citympg')
    highwaympg = request.form.get('highwaympg')

    # Categorical Inputs

    CarName = request.form.get('CarName')
    fueltype = request.form.get('fueltype')
    aspiration = request.form.get('aspiration')
    doornumber = request.form.get('doornumber')
    carbody = request.form.get('carbody')
    drivewheel = request.form.get('drivewheel')
    enginelocation = request.form.get('enginelocation')
    enginetype = request.form.get('enginetype')
    cylindernumber = request.form.get('cylindernumber')
    fuelsystem = request.form.get('fuelsystem')

    prediction = model.predict(
    pd.DataFrame(
        columns=['car_ID','symboling','CarName','fueltype','aspiration',
                 'doornumber','carbody','drivewheel','enginelocation',
                 'wheelbase','carlength','carwidth','carheight','curbweight',
                 'enginetype','cylindernumber','enginesize','fuelsystem',
                 'boreratio','stroke','compressionratio','horsepower',
                 'peakrpm','citympg','highwaympg'],

        data=np.array([car_ID,symboling,CarName,fueltype,aspiration,
                       doornumber,carbody,drivewheel,enginelocation,
                       wheelbase,carlength,carwidth,carheight,curbweight,
                       enginetype,cylindernumber,enginesize,fuelsystem,
                       boreratio,stroke,compressionratio,horsepower,
                       peakrpm,citympg,highwaympg]).reshape(1,25)
                )
    )

   

    print(prediction)

    return str(np.round(prediction[0],2))


if __name__ == "__main__":
    app.run()