from app.routes import db


class Match(db.Model):
    __tablename__ = 'matches'

    id = db.Column(db.Integer, primary_key=True)
    date = db.Column(db.Date())
    season_round = db.Column(db.String())
    venue = db.Column(db.String())
    home_score = db.Column(db.Integer)
    away_score = db.Column(db.Integer)

    home_team_id = db.Column(db.Integer, db.ForeignKey('teams.id'))
    home_team = db.relationship(
        'Team', back_populates='home_matches', foreign_keys='Match.home_team_id'
    )
    away_team_id = db.Column(db.Integer, db.ForeignKey('teams.id'))
    away_team = db.relationship(
        'Team', back_populates='away_matches', foreign_keys='Match.away_team_id'
    )

    home_betting_odds = db.relationship(
        'BettingOdds',
        back_populates='home_match',
        uselist=False,
        primaryjoin='Match.id==BettingOdds.home_match_id'
    )
    away_betting_odds = db.relationship(
        'BettingOdds',
        back_populates='away_match',
        uselist=False,
        primaryjoin='Match.id==BettingOdds.away_match_id'
    )

    def __repr__(self):
        return (
            f'<Match(date={self.date}, season_round={self.season_round}, ' +
            f'home_team={self.home_team}, away_team={self.away_team}, venue={self.venue}, ' +
            f'home_score={self.home_score}, away_score={self.away_score})>'
        )


class BettingOdds(db.Model):
    __tablename__ = 'betting_odds'

    id = db.Column(db.Integer, primary_key=True)
    win_odds = db.Column(db.Float())
    point_spread = db.Column(db.Integer)

    home_match_id = db.Column(db.Integer, db.ForeignKey('matches.id'))
    home_match = db.relationship(
        'Match', back_populates='home_betting_odds', foreign_keys='BettingOdds.home_match_id'
    )
    away_match_id = db.Column(db.Integer, db.ForeignKey('matches.id'))
    away_match = db.relationship(
        'Match', back_populates='away_betting_odds', foreign_keys='BettingOdds.away_match_id'
    )

    team = db.relationship('Team', back_populates='betting_odds')
    team_id = db.Column(db.Integer, db.ForeignKey('teams.id'))

    def __repr__(self):
        return (f'<BettingOdds(win_odds={self.win_odds}, point_spread={self.point_spread} ' +
                f'home_match={self.home_match}, away_match={self.away_match}, team={self.team})>')


class Team(db.Model):
    __tablename__ = 'teams'

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String())

    home_matches = db.relationship(
        'Match',
        back_populates='home_team',
        primaryjoin='Team.id==Match.home_team_id',
        lazy='noload'
    )
    away_matches = db.relationship(
        'Match',
        back_populates='away_team',
        primaryjoin='Team.id==Match.away_team_id',
        lazy='noload'
    )
    betting_odds = db.relationship('BettingOdds', back_populates='team', lazy='noload')

    def __repr__(self):
        return (f'<Team(name={self.name}, home_matches={self.home_matches}, ' +
                f'away_matches={self.away_matches}, betting_odds={self.betting_odds})>')
