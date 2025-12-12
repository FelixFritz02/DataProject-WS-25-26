import numpy as np
import time
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


class NestedCVRegressor:
    
    def __init__(self, model, param_grid, outer_splits=5, inner_splits=5):
        self.model = model
        self.param_grid = param_grid
        self.outer_splits = outer_splits
        self.inner_splits = inner_splits
        
        # Ergebnisse
        self.outer_mse = []
        self.outer_r2 = []
        self.best_params = []
        self.fit_times = []

    def run(self, X, y, output=False):
        """Führt Nested Cross Validation aus"""
        
        outer_cv = KFold(n_splits=self.outer_splits, shuffle=True, random_state=0)
        inner_cv = KFold(n_splits=self.inner_splits, shuffle=True, random_state=1)
        outer_fold = 0

        # ---- NEU: Datentyp feststellen ----
        X_is_df = isinstance(X, pd.DataFrame)
        y_is_series = isinstance(y, pd.Series)

        for train_ix, test_ix in outer_cv.split(X):
            outer_fold += 1

            # ----------- NEU: Pandas-kompatibles Indexing ------------
            if X_is_df:
                X_train = X.iloc[train_ix]
                X_test  = X.iloc[test_ix]
            else:
                X_train = X[train_ix]
                X_test  = X[test_ix]

            if y_is_series:
                y_train = y.iloc[train_ix].values
                y_test  = y.iloc[test_ix].values
            else:
                y_train = y[train_ix]
                y_test  = y[test_ix]
            # ---------------------------------------------------------

            # Inner CV + hyperparameter tuning
            if output:
                grid_search = GridSearchCV(
                    estimator=self.model,
                    param_grid=self.param_grid,
                    scoring="neg_mean_squared_error",
                    cv=inner_cv,
                    n_jobs=1,
                    verbose=2
                )
            else:
                grid_search = GridSearchCV(
                    estimator=self.model,
                    param_grid=self.param_grid,
                    scoring="neg_mean_squared_error",
                    cv=inner_cv,
                    n_jobs=-1
                )

            grid_search.fit(X_train, y_train)

            # beste hyperparameter sichern
            best_params = grid_search.best_params_
            self.best_params.append(best_params)

            # bestes Modell neu fitten (→ Messung dieser Zeit)
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
                print(
                    f"Outer Fold {outer_fold}/{self.outer_splits} | "
                    f"Best Params: {best_params} | "
                    f"Fit Time: {end - start:.3f}s | "
                    f"Outer R²: {self.outer_r2[-1]:.3f} | MSE: {self.outer_mse[-1]:.3f}"
                )

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

    def plot_scores(self, title=None):
        sns.set_theme(style="whitegrid")

        df = pd.DataFrame({
            "R²": self.outer_r2,
            "MSE": self.outer_mse,
            "Fit Time (s)": self.fit_times
        })

        df_long = df.melt(var_name="Metric", value_name="Value")

        g = sns.catplot(
            x="Metric",
            y="Value",
            col="Metric",
            data=df_long,
            kind="box",
            col_wrap=3,
            sharex=False,
            sharey=False,
            palette="Set2",
            height=5,
            aspect=0.8
        )

        g.set_axis_labels("", "Wert")
        plt.show()

        