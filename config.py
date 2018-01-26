import os


class Config(object):
    DEBUG = False
    TESTING = False
    DEVELOPMENT = False


class ProductionConfig(Config):
    DATABASE_URL = os.getenv('DATABASE_URL')


class DevelopmentConfig(Config):
    DEVELOPMENT = True
    DEBUG = True
    DATABASE_URL = "postgresql://localhost:5432/footy_tipper"


class TestingConfig(Config):
    DEBUG = True
    TESTING = True
