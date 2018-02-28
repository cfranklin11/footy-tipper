import os
import sys
from flask import Flask, render_template, abort, request, jsonify, make_response
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
    from app.actions.prepare_model_data import ModelData
    from app.actions.predict_results import MLModel

    if request.args.get('password') == app.config['PASSWORD']:
        X, y = ModelData(app.config['DATABASE_URL'], N_STEPS).prediction_data()
        predictions = MLModel(N_STEPS).predict(X, y)

        return jsonify(predictions)
    else:
        abort(401)


@app.route('/update_data', methods=['POST'])
def update():
    from app.actions.scrape_data import PageScraper
    from app.actions.clean_data import DataCleaner
    from app.actions.save_data import DataSaver

    if request.args.get('password') == app.config['PASSWORD']:
        data = PageScraper().data()
        dfs = DataCleaner(data).data()
        DataSaver(dfs).save_data()

        return make_response('New data was successfully saved.')
    else:
        abort(401)
