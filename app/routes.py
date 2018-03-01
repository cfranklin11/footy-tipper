import os
import sys
from datetime import datetime
from flask import Flask, render_template, abort, request
from flask_sqlalchemy import SQLAlchemy
from dotenv import load_dotenv
import sendgrid
from sendgrid.helpers.mail import Email, Content, Mail

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
app.config['DATABASE_URL'] = os.environ.get('DATABASE_URL')

app.config['SQLALCHEMY_DATABASE_URI'] = app.config['DATABASE_URL']
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

app.config['SENDGRID_API_KEY'] = os.environ.get('SENDGRID_API_KEY')
app.config['EMAIL_RECIPIENT'] = os.environ.get('EMAIL_RECIPIENT')

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

        sg = sendgrid.SendGridAPIClient(apikey=app.config['SENDGRID_API_KEY'])
        from_email = Email('predictions@footytipper.com')
        to_email = Email(app.config['EMAIL_RECIPIENT'])
        subject = f'Footy Tips for {datetime.now().date()}'
        content = Content("text/plain", str(predictions))
        mail = Mail(from_email, subject, to_email, content)
        response = sg.client.mail.send.post(request_body=mail.get())

        return (response.body, response.status_code, response.headers.items())
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

        return ('New data was successfully saved.', 200)
    else:
        abort(401)
