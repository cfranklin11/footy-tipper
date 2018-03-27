import numpy as np
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

import re
from tensorflow.python.keras import models, layers, backend as K, callbacks
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.utils.metaestimators import _BaseComposition
from sklearn.pipeline import make_pipeline
from vowpalwabbit.sklearn_vw import VWClassifier
import tempfile
from lib.models.preprocessing import (
    TimeStepReshaper,
    TimeStepOneHotEncoder,
    CategoryEncoder,
    ColumnFilter
)
from lib.models.model_data import FitDataCreator, PredictDataCreator

N_STEPS_REGEX = re.compile('_n_steps$')


class RNNBinaryClassifier(BaseEstimator, ClassifierMixin):
    """
    Wrapper class for a keras RNN classifier model.

    Parameters
    ----------
        n_teams : {int}
            Size of team vocab.
        n_years : {int}
            Number of years in data set.
        n_categories : {int}
            Total number of category features.
        n_features : {int}
            Total number of features of X input.

        n_steps : {int}
            Number of time steps (i.e. past observations) to include in the data.
            (This is the 2nd dimension of the input data.)
        patience : {int}
            Number of epochs of declining performance before model stops early.
        dropout : {float}, range = 0 to 1
            Percentage of data that's dropped from inputs.
        recurrent_dropout : {float}, range = 0 to 1
            Percentage of data that's dropped from recurrent state.
        cell_size : {int}
            Number of neurons per layer.
        team_dim : {int}
            Output dimension of embedded team data.
        batch_size : {int}
            Number of observations per batch (keras default is 32).
        team : {boolean}
            Whether to include teams input in model.
        oppo_team : {boolean}
            Whether to include oppo_teams input model.
        verbose : {int}, range = 1 to 3
            How frequently messages are printed during training.
        n_hidden_layers : {int}
            How many hidden layers to include in the model.
        epochs : {int}
            Max number of epochs to run during training.
    """

    def __init__(self, n_teams, n_years, n_venues, n_categories, n_features, n_steps=None,
                 patience=10, dropout=0.2, recurrent_dropout=0, cell_size=50, team_dim=5,
                 batch_size=None, team=True, oppo_team=True, last_finals_reached=True,
                 finals_dim=4, venue=True, venue_dim=4, verbose=0, n_hidden_layers=1,
                 epochs=200, optimizer='adam', activation='tanh',
                 recurrent_activation='hard_sigmoid', kernel_initializer='glorot_uniform'):
        if n_steps is None:
            raise Exception('n_steps is None. Be sure to explicitly set n_steps ' +
                            'to avoid data shape errors.')
        if type(cell_size) == list and len(cell_size) != n_hidden_layers + 1:
            raise Exception('cell_size must be an integer or a list with length' +
                            'equal to number of layers. cell_size has {} '.format(len(cell_size)) +
                            'values and there are {} layers in this model.'.format(n_hidden_layers + 1))

        self.n_teams = n_teams
        self.n_years = n_years
        self.n_venues = n_venues
        self.n_categories = n_categories
        self.n_features = n_features

        self.n_steps = n_steps
        self.patience = patience
        self.dropout = dropout
        self.recurrent_dropout = recurrent_dropout
        self.cell_size = cell_size
        self.team_dim = team_dim
        self.batch_size = batch_size
        self.team = team
        self.oppo_team = oppo_team
        self.last_finals_reached = last_finals_reached
        self.finals_dim = finals_dim
        self.venue = venue
        self.venue_dim = venue_dim
        self.verbose = verbose
        self.n_hidden_layers = n_hidden_layers
        self.epochs = epochs
        self.optimizer = optimizer
        self.activation = activation
        self.recurrent_activation = recurrent_activation
        self.kernel_initializer = kernel_initializer

        # We have to include the time reshaper/encoder in the model instead of
        # separate pipelines for consistency during parameter tuning.
        # Both the model and reshaper take n_steps as a parameter and must use
        # the same n_steps value.
        self.X_reshaper = TimeStepReshaper(n_steps=n_steps)
        self.y_reshaper = make_pipeline(
            TimeStepReshaper(n_steps=n_steps, are_labels=True),
            TimeStepOneHotEncoder(sparse=False)
        )

        self.__create_model()

    def fit(self, X, y):
        """
        Fit model to training data.

        Parameters
        ----------
            X : {array-like}, shape = [n_observations, n_features].
                Samples
            y : {array-like}, shape = [n_observations, 1]

        Returns
        -------
            self
        """

        self.model.fit(
            self.__inputs(X),
            self.__y(y),
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_split=0.2,
            verbose=self.verbose,
            # Using loss instead of accuracy, because it bounces around less.
            # Also, accuracy tends to reach its max a little after loss reaches its min,
            # meaning the early-stopping delay improves performance.
            callbacks=[callbacks.EarlyStopping(monitor='val_loss', patience=self.patience)]
        )

        return self

    def predict_proba(self, X):
        """
        Predict label-class probabilities for each observation.

        Parameters
        ----------
            X : {array-like}, shape = [n_observations, n_features].
                Samples

        Returns
        -------
            y : {array-like}, shape = [n_observations, n_classes]
                Predicted class probabilities.
        """

        return self.model.predict(self.__inputs(X))

    def predict(self, X):
        """
        Predict which class each observation belongs to.

        Parameters
        ----------
            X : {array-like}, shape = [n_observations, n_features].
                Samples

        Returns
        -------
            y : {array-like}, shape = [n_observations]
                Predicted class labels.
        """

        y_pred = self.predict_proba(X)
        return np.argmax(y_pred, axis=1) if y_pred.shape[1] > 1 else (y_pred[:, 0] >= 0.5).astype(int)

    def score(self, X, y):
        """
        Score the model based on the loss and metrics passed to the keras model.

        Parameters
        ----------
            X : {array-like}, shape = [n_observations, n_features].
                Samples
            y : {array-like}, shape = [n_observations, 1]

        Returns
        -------
            scores : {list}, [loss_score, metric_score]
        """

        return self.model.evaluate(
            self.__inputs(X),
            self.__y(y)
        )

    def set_params(self, **params):
        """
        Set params for this model instance, and create a new keras model if params have changed.

        Parameters
        ----------
            **params : {named parameters}

        Returns
        -------
            self
        """

        prev_params = self.get_params()
        # NOTE: Don't do an early return if params haven't changed, because it causes an
        # error when n_jobs > 1

        # Use parent set_params method to avoid infinite loop
        super(RNNBinaryClassifier, self).set_params(**params)

        # Only need to recreate reshapers if n_steps has changed
        if prev_params['n_steps'] != self.n_steps:
            self.X_reshaper.set_params(n_steps=self.n_steps)
            self.y_reshaper.set_params(timestepreshaper__n_steps=self.n_steps)

        # Clear session to avoid memory leak
        K.clear_session()

        # Need to recreate model after changing any relevant params
        self.__create_model()

        return self

    # Adapted this code from: http://zachmoshe.com/2017/04/03/pickling-keras-models.html
    # This is necessary for GridSearch/RansomSearch, because keras models can't
    # be pickled by default.
    def __getstate__(self):
        model_str = ""
        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=True) as f:
            models.save_model(self.model, f.name, overwrite=True)
            model_str = f.read()
        d = {key: value for key, value in self.__dict__.items() if key != 'model'}
        d.update({'model_str': model_str})
        return d

    def __setstate__(self, state):
        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=True) as f:
            f.write(state['model_str'])
            f.flush()
            model = models.load_model(f.name)
        d = {value: key for value, key in state.items() if key != 'model_str'}
        d.update({'model': model})
        self.__dict__ = d

    def __inputs(self, X):
        """
        Prepare X data array to fit expected input shapes for model, and create list
        of input arrays to pass to keras model methods.

        Parameters
        ----------
            X : {array-like}, shape = [n_observations, n_features].
                Samples

        Returns
        -------
            input_list : {list}, shape = [n_inputs, n_observations, n_steps, 1 or n_numeric_features]
        """

        reshaped_X = self.X_reshaper.fit_transform(X)
        cat_inputs = [reshaped_X[:, :, category_n] for category_n in range(self.n_categories)]
        num_input = [reshaped_X[:, :, self.n_categories:]]

        return cat_inputs + num_input

    def __y(self, y):
        """
        Prepare y data array to fit expected input shape for the model.

        Parameters
        ----------
            y : {array-like}, shape = [n_observations, 1]
        """

        reshaped_y = self.y_reshaper.fit_transform(y)
        return reshaped_y[:, 0, :]

    def __create_model(self):
        """
        Create the keras model.
        """
        if type(self.cell_size) == int:
            cell_size = [self.cell_size] * (self.n_hidden_layers + 1)
        else:
            cell_size = self.cell_size

        team_input = self.__category_input('team_input')
        oppo_team_input = self.__category_input('oppo_team_input')
        finals_input = self.__category_input('finals_input')
        venue_input = self.__category_input('venue_input')

        numeric_input = layers.Input(
            shape=(self.n_steps, self.n_features - self.n_categories),
            dtype='float32',
            name='numeric_input'
        )

        team_embed = self.__team_embedding_layer('embedding_team')(team_input)
        oppo_team_embed = self.__team_embedding_layer('embedding_oppo_team')(oppo_team_input)
        finals_embed = layers.Embedding(
            input_dim=10,
            output_dim=self.finals_dim,
            input_length=self.n_steps,
            name='embedding_finals'
        )(finals_input)
        venue_embed = layers.Embedding(
            input_dim=self.n_venues * 2,
            output_dim=self.venue_dim,
            input_length=self.n_steps,
            name='embedding_venue'
        )(venue_input)

        layers_list = []

        # Made including these inputs boolean parameters rather than having the input
        # length be variable, because coordinating inclusion/exclusion of columns
        # among transformers & model proved too difficult to implement correctly
        if self.team:
            layers_list.append(team_embed)
        if self.oppo_team:
            layers_list.append(oppo_team_embed)
        if self.last_finals_reached:
            layers_list.append(finals_embed)
        if self.venue:
            layers_list.append(venue_embed)
        if self.n_features > self.n_categories:
            layers_list.append(numeric_input)

        merged_layers = layers.concatenate(layers_list)
        lstm = layers.LSTM(
            cell_size[0],
            kernel_initializer=self.kernel_initializer,
            activation=self.activation,
            recurrent_activation=self.recurrent_activation,
            input_shape=(self.n_steps, self.n_features),
            dropout=self.dropout,
            recurrent_dropout=self.recurrent_dropout,
            return_sequences=self.n_hidden_layers > 0,
            name='lstm_1'
        )(merged_layers)

        # Allow for variable number of hidden layers, returning sequences to each
        # subsequent LSTM layer
        for idx, layer_n in enumerate(range(self.n_hidden_layers)):
            lstm = layers.LSTM(
                cell_size[idx + 1],
                kernel_initializer=self.kernel_initializer,
                activation=self.activation,
                recurrent_activation=self.recurrent_activation,
                dropout=self.dropout,
                recurrent_dropout=self.recurrent_dropout,
                return_sequences=layer_n < self.n_hidden_layers - 1,
                name='lstm_{}'.format(idx + 2)
            )(lstm)

        output = layers.Dense(2, activation='softmax')(lstm)

        self.model = models.Model(
            inputs=[
                team_input,
                oppo_team_input,
                finals_input,
                venue_input,
                numeric_input
            ],
            outputs=output
        )
        self.model.compile(
            loss='binary_crossentropy',
            optimizer=self.optimizer,
            metrics=['accuracy']
        )

    def __category_input(self, name):
        """
        Create a categroy input for the keras model.

        Parameters
        ----------
            name : {string}
                Name for the input. Must be unique.

        Returns
        -------
            category_input : {layers.input}
                1-dimension input for a category feature.
        """
        return layers.Input(shape=(self.n_steps,), dtype='int32', name=name)

    def __team_embedding_layer(self, name):
        """
        Create embedding layer for a team category feature.

        Returns
        -------
            team_embedding_layer : {layers.embedding}
                Embedding layer for team category.
        """
        return layers.Embedding(
            input_dim=self.n_teams * 2,
            output_dim=self.team_dim,
            input_length=self.n_steps,
            name=name
        )


class TimeStepBinaryClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, model, n_steps=None, **params):
        if n_steps is None:
            raise Exception('n_steps is None. Be sure to explicitly set n_steps ' +
                            'to avoid data shape errors.')
        self.n_steps = n_steps

        # We have to include the time reshaper/encoder in the model instead of
        # separate pipelines for consistency during parameter tuning.
        # Both the model and reshaper take n_steps as a parameter and must use
        # the same n_steps value.
        self.X_reshaper = TimeStepReshaper(n_steps=n_steps, rnn=False)
        self.y_reshaper = TimeStepReshaper(n_steps=n_steps, are_labels=True, rnn=False)

        # model_params = {key: value for key, value in params.items() if key != 'n_steps'}
        self.model = model
        self.model.set_params(**params)

    def fit(self, X, y):
        """
        Fit model to training data.

        Parameters
        ----------
            X : {array-like}, shape = [n_observations, n_features].
                Samples
            y : {array-like}, shape = [n_observations, 1]

        Returns
        -------
            self
        """

        self.model.fit(self.__X(X), self.__y(y))
        return self

    def predict_proba(self, X):
        """
        Predict label-class probabilities for each observation.

        Parameters
        ----------
            X : {array-like}, shape = [n_observations, n_features].
                Samples

        Returns
        -------
            y : {array-like}, shape = [n_observations, n_classes]
                Predicted class probabilities.
        """
        return self.model.predict_proba(self.__X(X, train=False))

    def predict(self, X):
        """
        Predict which class each observation belongs to.

        Parameters
        ----------
            X : {array-like}, shape = [n_observations, n_features].
                Samples

        Returns
        -------
            y : {array-like}, shape = [n_observations]
                Predicted class labels.
        """

        y_pred = self.predict_proba(X)
        return np.argmax(y_pred, axis=1) if y_pred.shape[1] > 1 else (y_pred[:, 0] >= 0.5).astype(int)

    def score(self, X, y):
        """
        Score the model based on the loss and metrics passed to the keras model.

        Parameters
        ----------
            X : {array-like}, shape = [n_observations, n_features].
                Samples
            y : {array-like}, shape = [n_observations, 1]

        Returns
        -------
            score : {float}
                Classification accuracy.
        """

        return self.model.score(self.__inputs(X), self.__y(y))

    # Added this so we can set model params directly (timestepbinaryclassifier__param)
    # instead of via timestepbinaryclassifier__model__param
    def set_params(self, n_steps=None, **params):
        if n_steps is not None:
            super(TimeStepBinaryClassifier, self).set_params(n_steps=n_steps)
            self.X_reshaper.set_params(n_steps=n_steps)
            self.y_reshaper.set_params(n_steps=n_steps)

        self.model.set_params(**params)
        return self

    def __X(self, X, train=True):
        """
        Prepare X data array to fit expected input shape for the model.

        Parameters
        ----------
            X : {array-like}, shape = [n_observations, n_features]
        """
        X_reshaped = self.X_reshaper.fit_transform(X) if train else self.X_reshaper.transform(X)
        return X_reshaped

    def __y(self, y, train=True):
        """
        Prepare y data array to fit expected input shape for the model.

        Parameters
        ----------
            y : {array-like}, shape = [n_observations, 1]
        """

        y_reshaped = self.y_reshaper.fit_transform(y) if train else self.y_reshaper.transform(y)
        return y_reshaped[:, 0]


class TimeStepVotingClassifier(_BaseComposition, ClassifierMixin, TransformerMixin):
    def __init__(self, estimators, weights=None):
        if weights is not None and len(estimators) != len(weights):
            raise(Exception('estimators ({}) and weights ({len(weights)}) '.format(len(estimators)) +
                            'arguments are different lengths.'))

        self.estimators = [
            self.__assign_estimators(estimator) for estimator in estimators
        ]
        self.weights = weights
        self._y_pipeline = make_pipeline(
            CategoryEncoder(columns=['team']),
            ColumnFilter(include='.*'),
        )

        # Make sure the voting classifier's n_steps and its estimators n_steps are
        # always consistent
        self.estimators_n_steps = self.__get_estimators_n_steps()

    def fit(self, X, y):
        """
        Fit model to training data.

        Parameters
        ----------
            X : {array-like}, shape = [n_observations, n_features].
                Samples
            y : {array-like}, shape = [n_observations, 1]

        Returns
        -------
            self
        """

        for idx, estimator in enumerate(self.estimators):
            n_steps = self.estimators_n_steps[idx]
            estimator_model = estimator[1]

            # Need to create X, y per estimator, because each can have a different number
            # of time steps, which requires different data shapes
            X_train, y_train, _ = FitDataCreator(n_steps).data(X, y)
            y_train_transformed = self._y_pipeline.fit_transform(y_train)

            estimator_model.fit(X_train, y_train_transformed)

        return self

    def predict_proba(self, X, full_year=True):
        """
        Predict label-class probabilities for each observation.

        Parameters
        ----------
            X : {array-like}, shape = [n_observations, n_features].
                Samples

        Returns
        -------
            y : {array-like}, shape = [n_observations, n_classes]
                Predicted class probabilities.
        """

        y_preds = []
        for idx, estimator in enumerate(self.estimators):
            n_steps = self.estimators_n_steps[idx]
            estimator_model = estimator[1]

            # Need to create X, y per estimator, because each can have a different number
            # of time steps, which requires different data shapes
            X_pred, _ = PredictDataCreator(n_steps, full_year=full_year).data(X)
            y_preds.append(estimator_model.predict_proba(X_pred))

        return np.average(y_preds, axis=0, weights=self.weights)

    def predict(self, X):
        """
        Predict which class each observation belongs to.

        Parameters
        ----------
            X : {array-like}, shape = [n_observations, n_features].
                Samples

        Returns
        -------
            y : {array-like}, shape = [n_observations]
                Predicted class labels.
        """

        y_pred = self.predict_proba(X)
        return np.argmax(y_pred, axis=1) if y_pred.shape[1] > 1 else (y_pred[:, 0] >= 0.5).astype(int)

    def set_params(self, **params):
        super(TimeStepVotingClassifier, self)._set_params('estimators', **params)

        # Make sure the voting classifier's n_steps and its estimators n_steps are
        # always consistent
        self.estimators_n_steps = self.__get_estimators_n_steps()

        return self

    def get_params(self, deep=True):
        """ Get the parameters of the VotingClassifier
        Parameters
        ----------
        deep: bool
            Setting it to True gets the various classifiers and the parameters
            of the classifiers as well
        """
        return super(TimeStepVotingClassifier, self)._get_params('estimators', deep=deep)

    def __assign_estimators(self, estimator):
        if type(estimator) == tuple:
            return estimator

        return (estimator.__class__.__name__.lower(), estimator)

    def __get_estimators_n_steps(self):
        return [
            self.__get_n_steps(idx, estimator) for idx, estimator in enumerate(self.estimators)
        ]

    def __get_n_steps(self, idx, estimator):
        # Get value of param key with 'n_steps' to account for multiple layers of
        # pipelines and estimators
        return next(
            (value for key, value in estimator[1].get_params().items()
             if N_STEPS_REGEX.search(key) is not None)
        )


class VWClassifierWrapper(VWClassifier):
    def fit(self, X, y=None, sample_weight=None):
        unique_labels = set(y)
        if len(unique_labels) == 2 and 0 in unique_labels and 1 in unique_labels:
            # Other Scikit Learn estimators expect binary classes to be 0 or 1
            y_transformed = np.array([-1 if label == 0 else 1 for label in y])
        else:
            y_transformed = y

        return super(VWClassifierWrapper, self).fit(X, y_transformed, sample_weight)

    def predict_proba(self, X):
        y_proba = super(VWClassifierWrapper, self).decision_function(X)

        # Sci-kit Learn classifiers expect matrix of probabilities for each class
        if len(y_proba.shape) == 1:
            # VW's decision function returns probability of true result
            return np.array([[1 - y, y] for y in y_proba])
        else:
            return y_proba

    def predict(self, X):
        y_pred = super(VWClassifierWrapper, self).predict(X)
        return np.array([0 if y == -1 else 1 for y in y_pred])

    # VW raises an error when joblib tries to pickle it, so we call its save & load methods
    # for custom pickling
    def __getstate__(self):
        model_str = ""
        with tempfile.NamedTemporaryFile(delete=True) as f:
            self.save(f.name)
            model_str = f.read()
        d = {'model_str': model_str}
        return d

    def __setstate__(self, state):
        with tempfile.NamedTemporaryFile(delete=True) as f:
            f.write(state['model_str'])
            f.flush()
            self.load(f.name)
