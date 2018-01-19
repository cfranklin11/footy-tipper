import os
from flask_script import Manager
from flask_migrate import Migrate, MigrateCommand
from app.routes import app, db
from app.models import Match, Team, BettingOdds


directory = os.path.abspath(os.path.join(os.path.dirname(__file__)))

app.config['CSRF_ENABLED'] = True
app.config.from_pyfile(os.path.join(directory, '.env'))
app.config['SQLALCHEMY_DATABASE_URI'] = app.config['DATABASE_URL']
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

migrate = Migrate(app, db)
manager = Manager(app)

manager.add_command('db', MigrateCommand)

if __name__ == '__main__':
    manager.run()
