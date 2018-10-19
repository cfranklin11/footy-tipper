# Footy Tipper

An ensemble machine-learning model for predicting the outcomes of AFL matches.

**2018 Markeplacer office footy tipping competitions champion!**

140 tips for the H&A season

## Setting up the app  in Docker
* Run app only: `docker run -p 5000:5000 -e PORT=5000 <image-name>`
* Create DB data tables: `docker-compose run python3 db/manage.py db upgrade`
* Seed data: `docker-compose run python3 db/seeds.py`

## Running tests in Docker
* `docker-compose run --rm web pytest`
