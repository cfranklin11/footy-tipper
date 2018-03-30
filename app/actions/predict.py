import os
import sys

project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if project_path not in sys.path:
    sys.path.append(project_path)

from app.routes import app
from app.actions.prepare_model_data import ModelData
from app.estimators.estimator import Estimator
from app.actions.send_mail import PredictionsMailer


class Predictor():
    def predict(self):
        X, y = ModelData(app.config['DATABASE_URL']).prediction_data()
        predictions = Estimator().predict(X, y)

        if app.config['PRODUCTION']:
            response = PredictionsMailer(
                app.config['SENDGRID_API_KEY']
            ).send(
                app.config['EMAIL_RECIPIENT'], str(predictions)
            )
            return response
        else:
            return predictions
