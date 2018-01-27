import os
import sys
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
if project_path not in sys.path:
    sys.path.append(project_path)

from app.routes import app
from app.models import Match, Team, BettingOdds
from db import seeds


def main():
    engine = create_engine(app.config['DATABASE_URL'])
    Session = sessionmaker(bind=engine)
    session = Session()

    # Due to model associations, must delete records in this order to avoid
    # foreign key errors.
    session.query(BettingOdds).delete()
    session.query(Match).delete()
    session.query(Team).delete()

    session.commit()
    session.close()

    seeds.main()


if __name__ == '__main__':
    main()
