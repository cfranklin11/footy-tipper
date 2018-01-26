import os
import sys
from flask import Flask, render_template, abort, request, jsonify
from flask_sqlalchemy import SQLAlchemy

project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
if project_path not in sys.path:
    sys.path.append(project_path)


N_STEPS = 5

app = Flask(__name__)
app.config['CSRF_ENABLED'] = True
app.config.from_pyfile('../.env')
app.config['SQLALCHEMY_DATABASE_URI'] = app.config['DATABASE_URL']
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():
    from app.middleware.ml_model import MatchData, ModelData, MLModel

    if request.args.get('password') == app.config['PASSWORD']:
        raw_data = MatchData(app.config['DATABASE_URL']).data()
        X, y = ModelData(N_STEPS).data(raw_data)
        predictions = MLModel(N_STEPS).predict(X, y)

        return jsonify(predictions)
    else:
        abort(401)
