import os
import sys
from flask import render_template
import pandas as pd

project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if project_path not in sys.path:
    sys.path.append(project_path)

from app.routes import app
from app.actions.prepare_model_data import ModelData
from app.estimators.estimator import Estimator
from app.actions.send_mail import PredictionsMailer


class Predictor():
    def predict(self):
        X, y, match_dates = ModelData(app.config['DATABASE_URL']).prediction_data()
        predictions = Estimator().predict(X, y)
        dated_predictions = (pd.concat([match_dates, predictions], axis=1)
                               .dropna()
                               .sort_values('full_date')
                               .to_dict('records'))

        if app.config['PRODUCTION']:
            self.__send_mail(dated_predictions)

        return dated_predictions

    def __send_mail(self, predictions):
        with app.app_context():
            (PredictionsMailer(app.config['SENDGRID_API_KEY'])
                .send(app.config['EMAIL_RECIPIENT'], self.__email_body(predictions)))

    def __email_body(self, predictions):
        return render_template('email.html', predictions=predictions)
