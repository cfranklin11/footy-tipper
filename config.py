class Config(object):
    DEBUG = False
    TESTING = False
    DEVELOPMENT = False
    PRODUCTION = False


class ProductionConfig(Config):
    PRODUCTION = True


class DevelopmentConfig(Config):
    DEVELOPMENT = True
    DEBUG = True


class TestingConfig(Config):
    DEBUG = True
    TESTING = True
