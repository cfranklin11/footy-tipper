import sys
import os
from sklearn.pipeline import make_pipeline, make_union
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib

project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if project_path not in sys.path:
    sys.path.append(project_path)

from app.middleware.ml_model import MatchData, CSVData
from lib.model.preprocessing import (
    CumulativeFeatureBuilder,
    CategoryEncoder,
    ColumnFilter,
    TimeStepDFCreator
)
from lib.model.model_data import create_model_data_sets

CATEGORY_REGEX = '^(?:oppo_)?team|venue|last_finals_reached'
N_STEPS = 5
TRAIN_START = 1966
# TODO: Keeping out a test year is only for developing prediction workflow.
# Final version will train on all available data.
TEST_YEAR = 2017
PARAMS = {
    'max_features': 0.967635,
    'min_samples_leaf': 5,
    'min_samples_split': 3,
    'n_estimators': 12,
    'random_state': 42
}


def main(db_url):
    db_df = MatchData(db_url).data()
    min_year = db_df['year'].min()
    csv_df = CSVData(os.path.join(project_path, 'data')).data(max_year=min_year - 1)
    df = csv_df.append(db_df).sort_index().fillna(0).drop('full_date', axis=1)

    numeric_pipeline = make_pipeline(ColumnFilter(exclude=CATEGORY_REGEX), StandardScaler())
    combined_features = make_union(
        ColumnFilter(include=CATEGORY_REGEX),
        numeric_pipeline,
    )

    X_train, _, y_train, _ = create_model_data_sets(
        df, n_steps=N_STEPS, train_start=TRAIN_START, test_years=(TEST_YEAR, 2017)
    )

    X_pipeline = make_pipeline(
        CumulativeFeatureBuilder(),
        CategoryEncoder(columns=['team', 'oppo_team', 'venue']),
        TimeStepDFCreator(n_steps=N_STEPS),
        combined_features
    )
    y_pipeline = make_pipeline(TimeStepDFCreator(n_steps=N_STEPS))

    model = RandomForestClassifier(**PARAMS)

    X_train_transformed = X_pipeline.fit_transform(X_train)
    y_train_transformed = y_pipeline.fit_transform(y_train)['win_t0']

    model.fit(X_train_transformed, y_train_transformed)

    joblib.dump(model, os.path.join(project_path, 'app/middleware/model.pkl'))
    joblib.dump(X_pipeline, os.path.join(project_path, 'app/middleware/X_pipeline.pkl'))
    joblib.dump(y_pipeline, os.path.join(project_path, 'app/middleware/y_pipeline.pkl'))


if __name__ == '__main__':
    db_url = len(sys.argv) > 1 and sys.argv[1]

    main(db_url)
