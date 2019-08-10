from flask import Flask, request, render_template
from sklearn.preprocessing import Normalizer
import numpy as np
import pandas as pd
import pickle

app = Flask(__name__)


dataset = pd.read_csv("data.csv")
X = dataset.iloc[:,2:-1].values

norm = Normalizer()
X = norm.fit_transform(X)

model = pickle.load(open("random_forest_after_balancing","rb"))

cols = [ 'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean',
     'smoothness_mean', 'compactness_mean', 'concavity_mean',
     'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
     'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
     'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',
     'fractal_dimension_se', 'radius_worst', 'texture_worst',
     'perimeter_worst', 'area_worst', 'smoothness_worst',
    'compactness_worst', 'concavity_worst', 'concave points_worst',
    'symmetry_worst', 'fractal_dimension_worst']

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/input")
def input():
    return render_template("input.html")

@app.route("/system_input_choice")
def system_input_choice():
    return render_template("system_input_choice.html")

@app.route("/system_input_malignant")
def system_input_malignant():
    return render_template("system_input_malignant.html")

@app.route("/system_input_benign")
def system_input_benign():
    return render_template("system_input_benign.html")

@app.route("/predict")
def predict():

    user_input = []
    
    for i in cols:
        user_input.append(float(request.args.get(i)))

    pred = model.predict(norm.transform(np.array([user_input])))

    return render_template("result.html", status = pred[0])
    
if __name__ == "__main__":
    app.run()