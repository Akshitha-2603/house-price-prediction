import numpy as np
import pandas as pd
from flask import Flask, render_template, request
import pickle


app = Flask(__name__)
data= pd.read_csv('Clean_data.csv')
pipe = pickle.load(open("C:\\Users\\akshi\\Desktop\\house\\lrmodel.pkl","rb"))


@app.route('/')
def index():

    locations = sorted(data['location'].unique())
    return render_template('houseprice.html', locations=locations)

@app.route('/predict', methods=['POST'])
def predict():
    location = request.form.get('location')
    bhk = request.form.get('bhk')
    bath = request.form.get('bath')
    ratepersqft=request.form.get('rate_persqft')
    areainsqft = request.form.get('area _insqft')
    buildingstatus=request.form.get('building_status')
    propertytype=request.form.get('property_type')
    constructionstatus=request.form.get('construction_status')


    print(location, bhk, bath,ratepersqft,areainsqft,buildingstatus,propertytype,constructionstatus)
    input = pd.DataFrame([[location,bath,bhk,ratepersqft,areainsqft,buildingstatus,propertytype,constructionstatus]],columns=['location', 'bath', 'bhk','rate_persqft','area _insqft','building_status','property_type','construction_status'])
    prediction = pipe.predict(input)[0]

    return str(np.round(prediction,2))

if __name__=="__main__":
    app.run(debug=True, port=5001)
