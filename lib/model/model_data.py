import pandas as pd


def create_model_data_sets(df, n_steps, train_start=2012, test_years=(2016, 2016), label='win'):
    train_years = (train_start, test_years[0] - 1)
    test_lead_in_years = (train_years[1], train_years[1])
    X_train = team_df(df, train_years).drop(label, axis=1)
    y_train = team_df(
        df[['team', 'year', 'round_number', label]], train_years
    ).drop(
        ['year', 'round_number'], axis=1
    )

    # We need the last n_steps rounds per team from previous year, so the validation data
    # will have the correct # of observations.
    test_lead_in_df = team_df(df, test_lead_in_years, start=-n_steps)
    test_df = df.xs(slice(*test_years), level=1, drop_level=False)
    full_test_df = pd.concat([test_lead_in_df, test_df]).sort_index()
    X_test = full_test_df.drop(label, axis=1)
    y_test = full_test_df[['team', label]]

    return X_train, X_test, y_train, y_test


def team_df(df, years, start=None, stop=None):
    unique_teams = df.index.get_level_values(0).drop_duplicates()

    return pd.concat(
        [
            df.xs(
                [team, slice(*years)], level=[0, 1], drop_level=False
                # Resetting index, because pandas flips out if I try to slice
                # with iloc while the df has a multiindex
            ).sort_index(
            ).reset_index(
                drop=True
            ).iloc[
                start:stop, :
            ] for team in unique_teams
        ]
    ).set_index(
        ['team', 'year', 'round_number'], drop=False
    ).sort_index()
