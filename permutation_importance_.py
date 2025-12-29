#aufruf der Funktion mit Validierungsdaten 80, 20 split und einem besten Modell
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.inspection import permutation_importance

def permutation_importance_analysis(model, X_val, y_val, save_as_pkl=False, file_name="", show_plot=False):
    # Berechnung der Permutations-Importanz
    perm_importance = permutation_importance(model, X_val, y_val)
    
    # Erstellung eines DataFrames zur besseren Visualisierung
    importance_df = pd.DataFrame({
        "feature": X_val.columns,
        "importance_mean": perm_importance.importances_mean,
        "importance_std": perm_importance.importances_std
    }).sort_values("importance_mean", ascending=True)
    if show_plot:
        plt.figure(figsize=(8, 5))

        plt.barh(
            importance_df["feature"],
            importance_df["importance_mean"],
            xerr=importance_df["importance_std"]
        )

        plt.xlabel("Permutation Importance (mean decrease in score)")
        plt.title("Permutation Feature Importance")
        plt.grid(True)

        plt.show()
    if save_as_pkl == True:
        importance_df.to_pickle("feature_importance/" + file_name)
    return importance_df