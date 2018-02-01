import os
import sys
from flask import Flask, render_template, abort, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from dotenv import load_dotenv

project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
if project_path not in sys.path:
    sys.path.append(project_path)

import config


N_STEPS = 5

app = Flask(__name__)

if os.getenv('PRODUCTION'):
    app.config.from_object(config.ProductionConfig)
else:
    app.config.from_object(config.DevelopmentConfig)
    dotenv_path = os.path.join(os.path.dirname(__file__), '../.env')
    load_dotenv(dotenv_path)

app.config['CSRF_ENABLED'] = True
app.config['PASSWORD'] = os.environ.get('PASSWORD')

app.config['SQLALCHEMY_DATABASE_URI'] = app.config['DATABASE_URL']
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():
    from app.middleware.ml_model import ModelData, MLModel

    if request.args.get('password') == app.config['PASSWORD']:
        X, y = ModelData(app.config['DATABASE_URL'], N_STEPS).prediction_data()
        predictions = MLModel(N_STEPS).predict(X, y)

        return jsonify(predictions)
    else:
        abort(401)
