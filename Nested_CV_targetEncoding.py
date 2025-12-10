import numpy as np
import time
import pandas as pd
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from category_encoders.target_encoder import TargetEncoder


class NestedCVRegressorWithTargetEncoding:

    def __init__(self, model, param_grid, encode_cols=None,
                 outer_splits=5, inner_splits=5, random_state=42, n_jobs=-1):
        
        self.model = model
        self.param_grid = param_grid
        self.encode_cols = encode_cols if encode_cols is not None else []
        self.outer_splits = outer_splits
        self.inner_splits = inner_splits
        self.random_state = random_state
        self.n_jobs = n_jobs
        
        self.results = {
            "mse": [],
            "r2": [],
            "fit_time": [],
            "best_params": [],
            "best_models": []
        }

    def _make_pipeline(self):
        """Erzeugt eine Pipeline mit TargetEncoder + Modell."""

        steps = []

        if self.encode_cols:
            steps.append(("encode", TargetEncoder(cols=self.encode_cols)))

        steps.append(("model", self.model))

        return Pipeline(steps)

    def run(self, X, y, verbose=False):

        # falls Pandas-DataFrame → für Indexing relevant
        if isinstance(X, pd.DataFrame):
            X_values = X
        else:
            X_values = pd.DataFrame(X)

        y_values = np.array(y)

        outer_cv = KFold(n_splits=self.outer_splits, shuffle=True, random_state=self.random_state)
        inner_cv = KFold(n_splits=self.inner_splits, shuffle=True, random_state=self.random_state + 1)

        for fold, (train_ix, test_ix) in enumerate(outer_cv.split(X_values), 1):

            X_train = X_values.iloc[train_ix]
            X_test  = X_values.iloc[test_ix]
            y_train = y_values[train_ix]
            y_test  = y_values[test_ix]

            # Pipeline erzeugen
            pipe = self._make_pipeline()

            # GridSearch über die Pipeline (Parameter für Modell mit "model__" prefix!)
            param_grid = {f"model__{k}": v for k, v in self.param_grid.items()}

            grid = GridSearchCV(
                estimator=pipe,
                param_grid=param_grid,
                scoring="neg_mean_squared_error",
                cv=inner_cv,
                n_jobs=self.n_jobs,
                verbose=2 if verbose else 0
            )

            grid.fit(X_train, y_train)

            best_params = grid.best_params_
            best_model = grid.best_estimator_

            start = time.perf_counter()
            best_model.fit(X_train, y_train)
            fit_time = time.perf_counter() - start

            y_pred = best_model.predict(X_test)

            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            self.results["mse"].append(mse)
            self.results["r2"].append(r2)
            self.results["fit_time"].append(fit_time)
            self.results["best_params"].append(best_params)
            self.results["best_models"].append(best_model)

            if verbose:
                print(f"[Fold {fold}/{self.outer_splits}] "
                      f"MSE={mse:.3f} | R²={r2:.3f} | Fit={fit_time:.2f}s | Best={best_params}")

    def get_results_df(self):
        return pd.DataFrame(self.results)

    def get_mean_scores(self):
        return {
            "mean_mse": np.mean(self.results["mse"]),
            "mean_r2": np.mean(self.results["r2"]),
            "mean_fit": np.mean(self.results["fit_time"])
        }

#Verwendung:
        """
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor()

param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [5, 10]
}

nested = NestedCVRegressorWithTargetEncoding(
    model=model,
    param_grid=param_grid,
    encode_cols=["state"]   # <<--- wichtige Zeile
)

nested.run(X_clean, X_clean["price"], verbose=True)

print(nested.get_results_df())
print(nested.get_mean_scores())

        """