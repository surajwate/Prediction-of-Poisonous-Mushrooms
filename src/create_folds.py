import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold

if __name__ == "__main__":
    df = pd.read_csv("./input/train.csv")
    df["kfold"] = -1
    df = df.sample(frac=1).reset_index(drop=True)
    y = df['class'].values
    skf = StratifiedKFold(n_splits=5)
    for fold, (train_idx, val_idx) in enumerate(skf.split(X=df, y=y)):
        df.loc[val_idx, 'kfold'] = fold
    df.to_csv("./input/train_folds.csv", index=False)
    