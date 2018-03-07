class Config(object):
    DEBUG = False
    TESTING = False
    DEVELOPMENT = False
    PRODUCTION = False
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    CSRF_ENABLED = True
    SQLALCHEMY_BINDS = {
        'test': 'sqlite://'
    }


class ProductionConfig(Config):
    PRODUCTION = True


class DevelopmentConfig(Config):
    DEVELOPMENT = True
    DEBUG = True


class TestingConfig(Config):
    DEBUG = True
    TESTING = True
