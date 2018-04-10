# README #

## Setting up the app  in Docker
* Run app only: `docker run -p 5000:5000 -e PORT=5000 <image-name>`
* Create DB data tables: `docker-compose run python3 db/manage.py db upgrade`
* Seed data: `docker-compose run python3 db/seeds.py`

## Running tests in Docker
* `docker-compose run -p 8000 --rm web pytest`
