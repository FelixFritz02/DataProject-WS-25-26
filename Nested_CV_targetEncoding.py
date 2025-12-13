import numpy as np
import time
import pandas as pd
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, QuantileTransformer
from category_encoders.target_encoder import TargetEncoder

class NestedCVRegressorWithTargetEncoding:

    def __init__(self, model, param_grid, encode_cols=None,
                 outer_splits=5, inner_splits=5, random_state=42, n_jobs=-1, scaler = StandardScaler):
        self.model = model
        self.param_grid = param_grid
        self.encode_cols = encode_cols if encode_cols is not None else []
        self.outer_splits = outer_splits
        self.inner_splits = inner_splits
        self.random_state = random_state
        self.n_jobs = n_jobs
        if scaler is not None:
            self.scaler = scaler()
        else:
            self.scaler = None

        # Ergebnisse
        self.outer_mse = []
        self.outer_r2 = []
        self.best_params = []
        self.fit_times = []

    def _make_pipeline(self):
        """Erstellt eine Pipeline mit optionalem TargetEncoder + Modell."""
        steps = []
        if self.encode_cols:
            steps.append(("encode", TargetEncoder(cols=self.encode_cols)))
        if self.scaler is not None:
            steps.append(("scaler", self.scaler))
        steps.append(("model", self.model))
        return Pipeline(steps)

    def filter_cities(self, X, threshold=0):
        """Rare Label Encoding für die 'City' Spalte."""
        X_filtered = X.copy()
        if 'cityname' in X_filtered.columns:
            city_counts = X_filtered['cityname'].value_counts()
            cities_to_keep = city_counts[city_counts >= threshold].index
            X_filtered['cityname'] = X_filtered['cityname'].apply(lambda x: x if x in cities_to_keep else 'Other')
        return X_filtered


    def run(self, X, y, output=False):
        """Führt Nested Cross Validation aus mit Target-Encoding für encode_cols"""

        # Pandas-Support
        if isinstance(X, pd.DataFrame):
            X_values = X
        else:
            X_values = pd.DataFrame(X)
        y_values = np.array(y)

        outer_cv = KFold(n_splits=self.outer_splits, shuffle=True, random_state=self.random_state)
        inner_cv = KFold(n_splits=self.inner_splits, shuffle=True, random_state=self.random_state + 1)

        outer_fold = 0
        for train_ix, test_ix in outer_cv.split(X_values):
            outer_fold += 1
            X_train, X_test = X_values.iloc[train_ix], X_values.iloc[test_ix]
            y_train, y_test = y_values[train_ix], y_values[test_ix]

            X_train = self.filter_cities(X_train)

            # Pipeline erzeugen
            pipe = self._make_pipeline()

            # Parameter für GridSearch (mit "model__" prefix)
            param_grid = {f"model__{k}": v for k, v in self.param_grid.items()}

            grid_search = GridSearchCV(
                estimator=pipe,
                param_grid=param_grid,
                scoring="neg_mean_squared_error",
                cv=inner_cv,
                n_jobs=self.n_jobs,
                verbose=2 if output else 0
            )

            # Inner CV + Hyperparameter tuning
            grid_search.fit(X_train, y_train)

            best_params = grid_search.best_params_
            self.best_params.append(best_params)

            best_model = grid_search.best_estimator_

            start = time.perf_counter()
            best_model.fit(X_train, y_train)
            end = time.perf_counter()
            self.fit_times.append(end - start)

            # Evaluate
            y_pred = best_model.predict(X_test)
            self.outer_mse.append(mean_squared_error(y_test, y_pred))
            self.outer_r2.append(r2_score(y_test, y_pred))

            if output:
                print(f"Outer Fold {outer_fold}/{self.outer_splits} | "
                      f"Best Params: {best_params} | "
                      f"Fit Time: {end - start:.3f}s | "
                      f"Outer R²: {self.outer_r2[-1]:.3f} | MSE: {self.outer_mse[-1]:.3f}")

    # Getter Methoden
    def get_mse_scores(self):
        return self.outer_mse

    def get_r2_scores(self):
        return self.outer_r2

    def get_best_params(self):
        return self.best_params

    def get_fit_times(self):
        return self.fit_times

    def get_mean_mse(self):
        return np.mean(self.outer_mse)

    def get_mean_r2(self):
        return np.mean(self.outer_r2)

    def get_mean_fit_time(self):
        return np.mean(self.fit_times)
