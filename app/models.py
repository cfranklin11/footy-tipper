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
            '<Match(date={}, season_round={}, '.format(self.date, self.season_round) +
            'home_team={}, away_team={}, venue={}, '.format(self.home_team, self.away_team, self.venue) +
            'home_score={}, away_score={})>'.format(self.home_score, self.away_score)
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

    def date(self):
        match = self.home_match or self.away_match
        return (match and match.date) or None

    def venue(self):
        match = self.home_match or self.away_match
        return (match and match.venue) or None

    def __repr__(self):
        return ('<BettingOdds(win_odds={}, point_spread={} '.format(self.win_odds, self.point_spread) +
                'home_match={}, away_match={}, team={})>'.format(self.home_match, self.away_match, self.team))


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
        return ('<Team(name={}, home_matches={}, '.format(self.name, self.home_matches) +
                'away_matches={}, betting_odds={})>'.format(self.away_matches, self.betting_odds))
