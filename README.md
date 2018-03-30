# README #

## Setting up the app  in Docker
* Create image and run app: `docker-compose up -d`
* Create DB data tables: `docker-compose run python3 db/manage.py db upgrade`
* Seed data: `docker-compose run python3 db/seeds.py`
