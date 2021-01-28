from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

from data import symptoms

model = pickle.load(open('model6.pkl', 'rb'))

app = Flask(__name__)


@app.route('/')
def main():
    return render_template('home.html', symptoms=symptoms)


@app.route('/predict', methods=['POST', 'GET'])
def home():

    data1 = request.form['symptom1']
    data2 = request.form['symptom2']
    data3 = request.form['symptom3']
    data4 = request.form['symptom4']
    data5 = request.form['symptom5']
    data6 = request.form['symptom6']
    data7 = request.form['symptom7']
    data8 = request.form['symptom8']
    data9 = request.form['symptom9']
    data10 = request.form['symptom10']
    data11 = request.form['symptom11']
    data12 = request.form['symptom12']
    data13 = request.form['symptom13']
    data14 = request.form['symptom14']
    data15 = request.form['symptom15']
    data16 = request.form['symptom16']
    data17 = request.form['symptom17']

    syms = [[data1, data2, data3, data4, data5, data6, data7,
             data8, data9, data10, data11, data12, data13, data14,
             data15, data16, data17]]

    columns_to_transform = ['symptom1', 'symptom2', 'symptom3', 'symptom4', 'symptom5',
                            'symptom6', 'symptom7', 'symptom8', 'symptom9', 'symptom10',
                            'symptom11', 'symptom12', 'symptom13', 'symptom14',
                            'symptom15', 'symptom16', 'symptom17']

    dataset = pd.DataFrame(data=syms, columns=columns_to_transform)

    dataset = __encoder(dataset, columns_to_transform)

    prediction = model.predict(dataset)
    return render_template('after.html', data=prediction)


def __encoder(df: pd.DataFrame, columns) -> pd.DataFrame:
    filtered_d = []
    for rows in df.values:
        for val in rows:
            if val:
                filtered_d.append(1)
            else:
                filtered_d.append(0)

    return pd.DataFrame([filtered_d], columns=columns)


if __name__ == "__main__":
    app.run(debug=False)
