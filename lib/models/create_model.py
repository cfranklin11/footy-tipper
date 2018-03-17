import numpy as np
np.random.seed(42)

import os
import sys
from datetime import datetime
import math
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.externals import joblib
import xgboost as xgb

# Need to add project path to sys path to use src modules
project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if project_path not in sys.path:
    sys.path.append(project_path)

from app.actions.prepare_model_data import CSVData, CumulativeFeatureBuilder
from lib.models.preprocessing import CategoryEncoder, ColumnFilter
from lib.models.model_wrappers import (
    RNNBinaryClassifier, TimeStepBinaryClassifier, TimeStepVotingClassifier, VWClassifierWrapper
)
from lib.models.model_data import FitDataCreator


MODEL_DIRECTORY = 'app/estimators/'
CATEGORY_REGEX = '^(?:oppo_)?team|last_finals_reached|venue'

BEST_PARAMS = {
    'rnnbinaryclassifier': {
        'cell_size': [80, 201, 176, 236, 158, 147, 209],
        'dropout': 0.775750,
        'n_hidden_layers': 6,
        'n_steps': 6,
        'patience': 10,
        'team_dim': 5
    },
    'adaboostclassifier': {
        'learning_rate': 0.013236,
        'n_estimators': 29,
        'n_steps': 3,
    },
    'gradientboostingclassifier': {
        'learning_rate': 0.037858,
        'max_depth': 1,
        'max_features': 0.520205,
        'min_samples_leaf': 3,
        'min_samples_split': 3,
        'n_estimators': 141,
        'n_steps': 2,
        'subsample': 0.672366
    },
    'randomforestclassifier': {
        'max_features': 0.660370,
        'min_samples_leaf': 5,
        'min_samples_split': 3,
        'n_estimators': 11,
        'n_steps': 6,
    },
    'vwclassifierwrapper': {
        'decay_learning_rate': 0.602902,
        'l': 0.807172,
        'l1': 0.457325,
        'l2': 0.145266,
        'n_steps': 5,
        'passes': 5
    },
    'xgbclassifier': {
        'colsample_bylevel': 0.775334,
        'colsample_bytree': 0.975836,
        'learning_rate': 0.104628,
        'max_depth': 4,
        'n_estimators': 118,
        'n_steps': 2,
        'reg_alpha': 0.554867,
        'reg_lambda': 1.318660,
        'subsample': 0.630721
    },
    'weights': [5.939276, 3.845337, 4.831269, 2.715953, 2.073309, 2.752764]
}


def main(project_path, train_start=None, train_end=None):
    print('\n\nStarted:', datetime.now().strftime('%H:%M:%S'), '\n')

    df = CSVData(os.path.join(project_path, 'data/')).data(max_year=math.inf)
    # Moved the CumulativeFeatureBuilder out of the pipeline, because it was slowing
    # down training, and tuning its params after a few passes doesn't make much of a difference
    cfb = CumulativeFeatureBuilder()
    df = cfb.transform(df.drop('win', axis=1), y=df['win'])
    voters = []
    model_params = {}

    data_train = df.xs(slice(train_start, train_end), level=1, drop_level=False)
    X_train = data_train.drop('win', axis=1)
    y_train = data_train[['team', 'year', 'round_number', 'win']]

    params = BEST_PARAMS['rnnbinaryclassifier']
    n_steps = params['n_steps']
    _, _, model_config = FitDataCreator(n_steps=params['n_steps']).data(X_train, y_train)

    models = [
        RNNBinaryClassifier(
            model_config['n_teams'],
            model_config['n_train_years'],
            model_config['n_venues'],
            model_config['n_categories'],
            model_config['n_features'],
            n_steps=n_steps
        ),
        AdaBoostClassifier(),
        GradientBoostingClassifier(),
        RandomForestClassifier(),
        VWClassifierWrapper(loss_function='logistic', link='logistic'),
        xgb.XGBClassifier()
    ]

    numeric_pipeline = make_pipeline(
        ColumnFilter(exclude=CATEGORY_REGEX),
        StandardScaler()
    )
    combined_features = make_union(
        ColumnFilter(include=CATEGORY_REGEX),
        numeric_pipeline,
    )

    for model in models:
        model_name = model.__class__.__name__.lower()
        params = BEST_PARAMS[model_name]
        n_steps = params['n_steps']

        if model_name == 'rnnbinaryclassifier':
            time_step_model = model
            param_middle = 'rnnbinaryclassifier'
        else:
            time_step_model = TimeStepBinaryClassifier(model, n_steps=n_steps)
            param_middle = 'timestepbinaryclassifier'

        X_pipeline = make_pipeline(
            CategoryEncoder(columns=['team', 'oppo_team', 'venue']),
            combined_features,
            time_step_model
        )

        voters.append((f'{model_name}_pipeline', X_pipeline))
        model_params = {
            **model_params,
            **{f'{model_name}_pipeline__{param_middle}__{key}': value
               for key, value in BEST_PARAMS[model_name].items()}
        }

    vc = TimeStepVotingClassifier(voters)
    vc.set_params(**model_params)

    model_name = vc.__class__.__name__
    first_year = train_start or X_train['year'].min()
    last_year = train_end or X_train['year'].max()
    print(f'\n\nTraining {model_name} on data from {first_year + 1} ' +
          f'(with lead-in data from {first_year}) to {last_year}\n')

    # y_train needs extra columns to allow for data reshaping with lead-in year
    # (in case of n_steps reshaping only, y_train just needs 'team' column)
    vc.fit(X_train, y_train[['team', 'year', 'round_number', 'win']])

    model_directory_path = os.path.join(project_path, MODEL_DIRECTORY)

    if not os.path.isdir(model_directory_path):
        os.path.makedirs(model_directory_path)

    joblib.dump(vc, os.path.join(
        model_directory_path,
        f'{model_name}_{str(datetime.now().date())}_{first_year}_to_{last_year}.pkl'
    ))

    print('\n\nFinished:', datetime.now().strftime('%H:%M:%S'), '\n')


if __name__ == '__main__':
    named_args = {}
    for arg in sys.argv[1:]:
        if '=' in arg:
            key_value = arg.split('=')

            try:
                named_args[key_value[0]] = int(key_value[1])
            except ValueError:
                named_args[key_value[0]] = key_value[1]

    if '--timeit' in sys.argv:
        import timeit
        print(timeit.timeit('main(project_path, **named_args)', globals=globals(), number=2))
    else:
        main(project_path, **named_args)
