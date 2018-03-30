import os
import sys
from datetime import datetime
import sendgrid
from sendgrid.helpers.mail import Email, Content, Mail

project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
if project_path not in sys.path:
    sys.path.append(project_path)


EMAIL_FROM = 'predictions@footytipper.com'


class PredictionsMailer():
    def __init__(self, api_key):
        self.client = sendgrid.SendGridAPIClient(apikey=api_key).client

    def send(self, email_recipient, content):
        from_email = Email(EMAIL_FROM)
        to_email = Email(email_recipient)
        subject = 'Footy Tips for {}'.format(datetime.now().date())
        content = Content("text/plain", content)
        mail = Mail(from_email, subject, to_email, content)

        return self.client.mail.send.post(request_body=mail.get())
