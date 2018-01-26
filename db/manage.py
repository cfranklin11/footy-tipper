import os
import sys
from flask_script import Manager
from flask_migrate import Migrate, MigrateCommand

directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
if directory not in sys.path:
    sys.path.append(directory)

from app.routes import app, db
from app.models import Match, Team, BettingOdds


migrate = Migrate(app, db)
manager = Manager(app)

manager.add_command('db', MigrateCommand)

if __name__ == '__main__':
    manager.run()
