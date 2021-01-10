from flask import Flask, request, jsonify, render_template
import numpy as np
import joblib
import pandas as pd
import json
from sklearn import preprocessing
from sklearn.metrics import classification_report, accuracy_score

#Our model
with open("final_model.joblib", "rb") as f:
    final_model = joblib.load(f)

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')
    
@app.route('/predict',methods=['POST'])
def predict():

    features = [float(x) for x in request.form.values()]
    final_features = [np.array(features)]
    prediction = final_model.predict(final_features)
    league_pred = prediction[0]

    return render_template("index.html", prediction_text='Your predicted League Index is League {}'.format(league_pred))

@app.route('/results',methods=['POST'])
def results():

    data = request.get_json(force=True)
    prediction = final_model.predict([np.array(list(data.values()))])

    league_pred = prediction[0]
    return jsonify(league_pred)

    
    
if __name__ == '__main__':
    app.run(debug=True, port=5000)