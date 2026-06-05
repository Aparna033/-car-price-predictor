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

     # 2. Categorical Inputs
    CarName = request.form.get('CarName')

    if CarName == "Select Car Name":
        return "Please select a valid car"
    selected_car = car[car['CarName'] == CarName].iloc[0]

    fueltype = request.form.get('fueltype')
    aspiration = request.form.get('aspiration')
    doornumber = request.form.get('doornumber')
    carbody = request.form.get('carbody')
    drivewheel = request.form.get('drivewheel')
    enginelocation = request.form.get('enginelocation')
    enginetype = request.form.get('enginetype')
    cylindernumber = request.form.get('cylindernumber')
    fuelsystem = request.form.get('fuelsystem')
 
    # 1. Numeric Inputs (Properly Casted)
    car_ID = int(selected_car['car_ID'])
    symboling = int(selected_car['symboling'])

    wheelbase = float(selected_car['wheelbase'])
    carlength = float(selected_car['carlength'])
    carwidth = float(selected_car['carwidth'])
    carheight = float(selected_car['carheight'])

    curbweight = int(selected_car['curbweight'])

    enginesize = int(selected_car['enginesize'])

    boreratio = float(selected_car['boreratio'])
    stroke = float(selected_car['stroke'])

    compressionratio = float(selected_car['compressionratio'])

    horsepower = int(selected_car['horsepower'])

    peakrpm = int(selected_car['peakrpm'])

    citympg = int(selected_car['citympg'])
    highwaympg = int(selected_car['highwaympg'])
    # 3. Create the DataFrame using a standard Python list of lists [[ ... ]]
    # This prevents pandas/numpy from forcing your numbers into strings!
    input_df = pd.DataFrame(
        data=[[
            car_ID, symboling, CarName, fueltype, aspiration,
            doornumber, carbody, drivewheel, enginelocation,
            wheelbase, carlength, carwidth, carheight, curbweight,
            enginetype, cylindernumber, enginesize, fuelsystem,
            boreratio, stroke, compressionratio, horsepower,
            peakrpm, citympg, highwaympg
        ]],
        columns=[
            'car_ID', 'symboling', 'CarName', 'fueltype', 'aspiration',
            'doornumber', 'carbody', 'drivewheel', 'enginelocation',
            'wheelbase', 'carlength', 'carwidth', 'carheight', 'curbweight',
            'enginetype', 'cylindernumber', 'enginesize', 'fuelsystem',
            'boreratio', 'stroke', 'compressionratio', 'horsepower',
            'peakrpm', 'citympg', 'highwaympg'
        ]
    )

    # 4. Predict
    prediction = model.predict(input_df)[0]
    print("Prediction:", prediction)

    if prediction < 0:
        return "Cannot predict the price"
    
    return str(round(prediction, 2))


if __name__ == "__main__":
    app.run()
