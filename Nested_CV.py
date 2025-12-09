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
        self.fit_times = []   # <-- Neue Liste

    def run(self, X, y, output = False):
        """Führt Nested Cross Validation aus"""
        
        outer_cv = KFold(n_splits=self.outer_splits, shuffle=True, random_state=0)
        inner_cv = KFold(n_splits=self.inner_splits, shuffle=True, random_state=1)
        outer_fold = 0
        for train_ix, test_ix in outer_cv.split(X):
            outer_fold = outer_fold +1
            if output == True:
                grid_search = GridSearchCV(
                    estimator=self.model,
                    param_grid=self.param_grid,
                    scoring="neg_mean_squared_error",
                    cv=inner_cv,
                    n_jobs=1,
                    verbose = 2
                )
            else:
                grid_search = GridSearchCV(
                    estimator=self.model,
                    param_grid=self.param_grid,
                    scoring="neg_mean_squared_error",
                    cv=inner_cv,
                    n_jobs=-1,

                )
            X_train, X_test = X[train_ix], X[test_ix]
            y_train, y_test = y[train_ix], y[test_ix]

            # Inner CV + hyperparameter tuning
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
            # Statusmeldung pro Outer-Fold
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
        """ Zeit des Modells welches für den betrachteten Outer Fold die besten besten Parameter zeigt"""
        return self.fit_times

    def get_mean_mse(self):
        return np.mean(self.outer_mse)

    def get_mean_r2(self):
        return np.mean(self.outer_r2)

    def get_mean_fit_time(self):
        return np.mean(self.fit_times)

    def plot_scores(self, title=None):
        sns.set_theme(style="whitegrid")

        # DataFrame wie gehabt erstellen (angenommen, die Listen sind bereits da)
        df = pd.DataFrame({
            "R²": self.outer_r2,
            "MSE": self.outer_mse,
            "Fit Time (s)": self.fit_times
        })

        # Wide → Long Format für Seaborn
        df_long = df.melt(var_name="Metric", value_name="Value")

        # Verwenden von sns.catplot mit 'col' und 'free_y'
        g = sns.catplot(
            x="Metric",
            y="Value",
            col="Metric",         # Erstellt für jede "Metric" eine separate Spalte (Facet)
            data=df_long,
            kind="box",           # Stellt den Plot als Boxplot dar
            col_wrap=3,           # Zeigt alle 3 Plots in einer Zeile
            sharex=False,         # Wichtig: Jede Achse ist unabhängig
            sharey=False,         # Wichtig: Jede Achse hat eine freie Skala
            palette="Set2",
            height=5,             # Höhe jedes Unterplots
            aspect=0.8
        )

        # Passen Sie die Layouts an, um Überlappungen zu vermeiden
        g.set_axis_labels("", "Wert")      # Entfernt die x-Achsenbeschriftung und vereinfacht die y-Beschriftung
    
        plt.show()
        