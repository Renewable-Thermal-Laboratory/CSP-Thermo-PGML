import os
import glob
import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

DATA_DIR = "data/processed"
TARGET_COL = "TC1_tip"
PHYSICAL_FEATURES = ["h", "flux", "abs", "surf"]

def load_dataset():
    files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    dfs = [pd.read_csv(f) for f in files]
    return pd.concat(dfs, ignore_index=True)

def fisher_score(X, y):
    """
    Fisher Score for regression: |mean_diff| / (std_X + std_y)
    """
    scores = {}
    for col in X.columns:
        mean_diff = np.abs(X[col].mean() - y.mean())
        score = mean_diff / (X[col].std() + y.std())
        scores[col] = score
    return scores

def rank_physical_features(df):
    df = df.dropna(subset=PHYSICAL_FEATURES + [TARGET_COL])
    X = df[PHYSICAL_FEATURES]
    y = df[TARGET_COL]

    # Normalize features
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=PHYSICAL_FEATURES)

    #Correlation
    corr_scores = X_scaled.corrwith(y).abs().sort_values(ascending=False)

    #Mutual Information
    mi_scores = mutual_info_regression(X_scaled, y)
    mi_scores = pd.Series(mi_scores, index=PHYSICAL_FEATURES).sort_values(ascending=False)

    #Fisher Score
    fisher_scores = fisher_score(X_scaled, y)
    fisher_scores = pd.Series(fisher_scores).sort_values(ascending=False)

    return corr_scores, mi_scores, fisher_scores

def plot_scores(corr, mi, fisher):
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    corr.plot(kind="bar", ax=axs[0], title="Correlation")
    mi.plot(kind="bar", ax=axs[1], title="Mutual Information")
    fisher.plot(kind="bar", ax=axs[2], title="Fisher Score")
    plt.tight_layout()
    plt.savefig("results_trial_3/features.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    df = load_dataset()
    corr_scores, mi_scores, fisher_scores = rank_physical_features(df)

    print("\nTop Physical Features by Correlation:")
    print(corr_scores)

    print("\nTop Physical Features by Mutual Information:")
    print(mi_scores)

    print("\nTop Physical Features by Fisher Score:")
    print(fisher_scores)

    plot_scores(corr_scores, mi_scores, fisher_scores)
