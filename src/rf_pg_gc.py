import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from clean import RareCategoryReplacer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import matthews_corrcoef, make_scorer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
import time
import gc
from analyze import missing_values

def run(fold):
    print(f"Training model with fold={fold}")
    df = pd.read_csv("./input/train_folds.csv")

    # Reduce memory footprint
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='float')

    for col in df.select_dtypes(include=['int64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='integer')

    print("Starting preprocessing...")
    pp_time = time.time()
    missing_values_summary = missing_values(df)
    missing_values_summary = missing_values_summary[
        missing_values_summary["Missing Percentage"] > 50
    ]
    df = df.drop(missing_values_summary.index, axis=1)

    train = df[df.kfold != fold].reset_index(drop=True)
    valid = df[df.kfold == fold].reset_index(drop=True)

    X_train = train.drop("class", axis=1)
    X_valid = valid.drop("class", axis=1)
    y_train = train["class"]
    y_valid = valid["class"]

    cat_cols = X_train.select_dtypes(include='object').columns
    num_cols = X_train.select_dtypes(exclude='object').columns

    imputer = SimpleImputer(strategy='median')
    X_train[num_cols] = imputer.fit_transform(X_train[num_cols])
    X_valid[num_cols] = imputer.transform(X_valid[num_cols])

    imputer = SimpleImputer(strategy='most_frequent')
    X_train[cat_cols] = imputer.fit_transform(X_train[cat_cols])
    X_valid[cat_cols] = imputer.transform(X_valid[cat_cols])

    replacer = RareCategoryReplacer(columns=cat_cols, proportion_threshold=0.01)
    X_train = replacer.fit_transform(X_train)
    X_valid = replacer.transform(X_valid)

    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    X_train_encoded = encoder.fit_transform(X_train[cat_cols])
    X_valid_encoded = encoder.transform(X_valid[cat_cols])

    X_train_encoded = pd.DataFrame(X_train_encoded, columns=encoder.get_feature_names_out(cat_cols))
    X_valid_encoded = pd.DataFrame(X_valid_encoded, columns=encoder.get_feature_names_out(cat_cols))

    X_train = X_train.drop(cat_cols, axis=1)
    X_valid = X_valid.drop(cat_cols, axis=1)

    X_train = pd.concat([X_train, X_train_encoded], axis=1)
    X_valid = pd.concat([X_valid, X_valid_encoded], axis=1)

    scaler = StandardScaler()
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_valid[num_cols] = scaler.transform(X_valid[num_cols])
    print("Preprocessing done.")
    pp_time = time.time() - pp_time
    print(f"Preprocessing took {pp_time:.2f} seconds")

    gc.collect()  # Manually trigger garbage collection

    start_time = time.time()

    model = RandomForestClassifier(n_jobs=-1)

    param_distributions = {
        'n_estimators': [100, 200, 500],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'max_features': ['auto', 'sqrt']
    }

    mcc_scorer = make_scorer(matthews_corrcoef)

    random_search = RandomizedSearchCV(estimator=model, param_distributions=param_distributions, 
                                       n_iter=30, cv=3, scoring=mcc_scorer, random_state=42, n_jobs=-1)

    random_search.fit(X_train, y_train)

    best_model = random_search.best_estimator_

    y_pred = best_model.predict(X_valid)

    mcc = matthews_corrcoef(y_valid, y_pred)
    print(f"Fold={fold}, MCC={mcc}")
    print(f"Best parameters found: {random_search.best_params_}")
    print(f"Best MCC score: {random_search.best_score_}")

    end_time = time.time()

    elapsed_time = end_time - start_time

    print(f"Model building took {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    run(0)
