from app.routes import db


class Match(db.Model):
    __tablename__ = 'matches'

    id = db.Column(db.Integer, primary_key=True)
    date = db.Column(db.DateTime())
    season_round = db.Column(db.String())
    home_team_id = db.Column(db.Integer, db.ForeignKey('teams.id'))
    home_team = db.relationship('Team', back_populates='home_matches')
    away_team_id = db.Column(db.Integer, db.ForeignKey('teams.id'))
    away_team = db.relationship('Team', back_populates='away_matches')
    venue = db.Column(db.String())
    crowd_size = db.Column(db.Integer)
    home_score = db.Column(db.Integer)
    away_score = db.Column(db.Integer)
    betting_odds_id = db.Column(db.Integer, db.ForeignKey('betting_odds.id'))
    betting_odds = db.relationship('BettingOdds', back_populates='match')

    def __init__(self, date, season_round, home_team, away_team, venue,
                 crowd_size, home_score, away_score):
        self.date = date
        self.season_round = season_round
        self.home_team = home_team
        self.away_team = away_team
        self.venue = venue
        self.crowd_size = crowd_size
        self.home_score = home_score
        self.away_score = away_score

    def __repr__(self):
        return (
            f'<Match(date={self.date}, season_round={self.season_round}, home_team={self.home_team}, ' +
            f'away_team={self.away_team}, venue={self.venue}, ' +
            f'crowd_size={self.crowd_size}, home_score={self.crowd_size}, away_score={self.away_score})>'
        )


class BettingOdds(db.Model):
    __tablename__ = 'betting_odds'

    id = db.Column(db.Integer, primary_key=True)
    match = db.relationship('Match', back_populates='betting_odds', uselist=False)
    home_win_odds = db.Column(db.Float())
    away_win_odds = db.Column(db.Float())
    home_point_spread = db.Column(db.Integer)
    away_point_spread = db.Column(db.Integer)

    def __init__(self, home_win_odds, away_win_odds, home_point_spread, away_point_spread):
        self.home_win_odds = home_win_odds
        self.away_win_odds = away_win_odds
        self.home_point_spread = home_point_spread
        self.away_point_spread = away_point_spread

    def __repr__(self):
        return (
            f'<BettingOdds(home_win_odds={self.home_win_odds}, ' +
            f'away_win_odds={self.away_win_odds}, home_point_spread={self.home_point_spread}, ' +
            f'away_point_spread{self.away_point_spread})>'
        )


class Team(db.Model):
    __tablename__ = 'teams'

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String())
    home_matches = db.relationship('Match', back_populates='home_team')
    away_matches = db.relationship('Match', back_populates='away_team')

    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return f'<Team(name={self.name})>'
