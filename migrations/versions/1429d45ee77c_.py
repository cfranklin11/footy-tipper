"""empty message

Revision ID: 1429d45ee77c
Revises: ecff8fe7e0d4
Create Date: 2018-01-21 13:43:22.455385

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '1429d45ee77c'
down_revision = 'ecff8fe7e0d4'
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_constraint('matches_away_betting_odds_id_fkey', 'matches', type_='foreignkey')
    op.drop_constraint('matches_home_betting_odds_id_fkey', 'matches', type_='foreignkey')
    op.drop_column('matches', 'home_betting_odds_id')
    op.drop_column('matches', 'away_betting_odds_id')
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.add_column('matches', sa.Column('away_betting_odds_id', sa.INTEGER(), autoincrement=False, nullable=True))
    op.add_column('matches', sa.Column('home_betting_odds_id', sa.INTEGER(), autoincrement=False, nullable=True))
    op.create_foreign_key('matches_home_betting_odds_id_fkey', 'matches', 'betting_odds', ['home_betting_odds_id'], ['id'])
    op.create_foreign_key('matches_away_betting_odds_id_fkey', 'matches', 'betting_odds', ['away_betting_odds_id'], ['id'])
    # ### end Alembic commands ###
