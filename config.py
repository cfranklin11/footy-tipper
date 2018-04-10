class Config(object):
    DEBUG = False
    TESTING = False
    DEVELOPMENT = False
    PRODUCTION = False
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    CSRF_ENABLED = True


class ProductionConfig(Config):
    PRODUCTION = True


class DevelopmentConfig(Config):
    DEVELOPMENT = True
    DEBUG = True


class TestingConfig(Config):
    TESTING = True
    CSRF_ENABLED = False
    DEBUG = False
