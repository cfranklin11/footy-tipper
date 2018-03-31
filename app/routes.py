import os
import sys
from flask import Flask, render_template, abort, request
from flask_sqlalchemy import SQLAlchemy
from dotenv import load_dotenv
from rq import Queue

project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
if project_path not in sys.path:
    sys.path.append(project_path)

from app.worker import conn
import config

app = Flask(__name__)

if os.getenv('PRODUCTION'):
    app.config.from_object(config.ProductionConfig)
elif os.getenv('TESTING'):
    app.config.from_object(config.TestingConfig)
    dotenv_path = os.path.join(os.path.dirname(__file__), '../.env.test')
    load_dotenv(dotenv_path)
else:
    app.config.from_object(config.DevelopmentConfig)
    dotenv_path = os.path.join(os.path.dirname(__file__), '../.env')
    load_dotenv(dotenv_path)

app.config['PASSWORD'] = os.environ.get('PASSWORD')
app.config['DATABASE_URL'] = os.environ.get('DATABASE_URL')
app.config['SQLALCHEMY_DATABASE_URI'] = app.config['DATABASE_URL']

app.config['SENDGRID_API_KEY'] = os.environ.get('SENDGRID_API_KEY')
app.config['EMAIL_RECIPIENT'] = os.environ.get('EMAIL_RECIPIENT')

db = SQLAlchemy(app)
q = Queue(connection=conn)


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():
    from flask import jsonify
    from app.actions.predict import Predictor

    if request.args.get('password') == app.config['PASSWORD']:
        if app.config['PRODUCTION']:
            job = q.enqueue(Predictor().predict)
            return ('Prediction job #{} is in queue. '.format(job.get_id()) +
                    'You will receive an e-mail shortly.\n', 200)
        else:
            predictions = Predictor().predict()
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

        return ('New data was successfully saved.\n', 200)
    else:
        abort(401)
