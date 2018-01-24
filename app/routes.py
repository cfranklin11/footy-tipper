from flask import Flask, render_template
from flask_sqlalchemy import SQLAlchemy


app = Flask(__name__)
app.config['CSRF_ENABLED'] = True
app.config.from_pyfile('../.env')
app.config['SQLALCHEMY_DATABASE_URI'] = app.config['DATABASE_URL']
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)


@app.route('/')
def home():
    return render_template('home.html')


if __name__ == '__main__':
    app.run()
