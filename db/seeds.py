import sys
import os
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
if project_path not in sys.path:
    sys.path.append(project_path)

from app.routes import app
from app.models import Match, Team, BettingOdds


def main():
    engine = create_engine(app.config['DATABASE_URL'])
    Session = sessionmaker(bind=engine)
    session = Session()

    match_df = pd.read_csv(
        os.path.join(project_path, 'data/ft_match_list.csv'),
        parse_dates=['full_date'],
        infer_datetime_format=True
    )

    team_names = match_df['home_team'].append(match_df['away_team']).drop_duplicates()
    teams = [Team(name=team_name) for team_name in team_names]
    session.add_all(teams)

    saved_teams = session.query(Team).all()
    match_records = match_df.to_dict('records')

    for match_record in match_records:
        match = {
            'date': match_record['full_date'],
            'season_round': match_record['season_round'],
            'venue': match_record['venue'],
            'home_score': match_record['home_score'],
            'away_score': match_record['away_score'],
            'home_team': next(
                team for team in saved_teams if team.name == match_record['home_team']
            ),
            'away_team': next(
                team for team in saved_teams if team.name == match_record['away_team']
            )
        }
        session.add(Match(**match))

    saved_matches = session.query(Match).all()
    betting_df = pd.read_csv(
        os.path.join(project_path, 'data/afl_betting.csv'),
        parse_dates=['full_date'],
        infer_datetime_format=True
    )
    betting_records = betting_df.to_dict('records')

    for betting_record in betting_records:
        betting_team = next(
            team for team in saved_teams if team.name == betting_record['team']
        )
        betting_match = next(
            (match for match in saved_matches
                if match.date == betting_record['full_date'].to_pydatetime() and
                match.venue == betting_record['venue'])
        )

        betting = {
            'win_odds': betting_record['win_odds'],
            'point_spread': betting_record['point_spread'],
            'team': betting_team
        }

        if betting_team.id == betting_match.home_team_id:
            betting['home_match'] = betting_match
        elif betting_team.id == betting_match.away_team_id:
            betting['away_match'] = betting_match
        else:
            raise(f'Betting data {betting_record} does not match any existing ' +
                  'team/match combinations')

        session.add(BettingOdds(**betting))

    try:
        session.commit()
    except:
        print('Something went wrong, rolling back')
        session.rollback()
        raise
    finally:
        print('Closing session...')
        session.close()


if __name__ == '__main__':
    main()
