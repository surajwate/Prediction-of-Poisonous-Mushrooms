# This script trains a Random Forest model using GridSearchCV to find the best hyperparameters
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from clean import RareCategoryReplacer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import matthews_corrcoef, accuracy_score
from analyze import missing_values
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import matthews_corrcoef, make_scorer
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
import time


def run(fold):
    # Load the training data with folds
    df = pd.read_csv("./input/train_folds.csv")
    df = df.sample(n=30000, random_state=42)

    # drop the columns with higher than 40% missing values
    missing_values_summary = missing_values(df)
    missing_values_summary = missing_values_summary[
        missing_values_summary["Missing Percentage"] > 50
    ]
    # df = df.drop(missing_values_summary.index, axis=1)

    # Split the data into training and validation sets
    train = df[df.kfold != fold].reset_index(drop=True)
    valid = df[df.kfold == fold].reset_index(drop=True)

    # Split the features and target
    X_train = train.drop("class", axis=1)
    X_valid = valid.drop("class", axis=1)
    y_train = train["class"]
    y_valid = valid["class"]

    # Get the categorical and numerical columns
    cat_cols =X_train.select_dtypes(include='object').columns
    num_cols = X_train.select_dtypes(exclude='object').columns

    # Fill missing values in the numerical columns with the median value
    imputer = SimpleImputer(strategy='median')
    X_train[num_cols] = imputer.fit_transform(X_train[num_cols])
    X_valid[num_cols] = imputer.transform(X_valid[num_cols])

    # Fill missing values in the categorical columsn with the mode value
    imputer = SimpleImputer(strategy='most_frequent')
    X_train[cat_cols] = imputer.fit_transform(X_train[cat_cols])
    X_valid[cat_cols] = imputer.transform(X_valid[cat_cols])

    replacer = RareCategoryReplacer(columns=cat_cols, proportion_threshold=0.01)
    X_train = replacer.fit_transform(X_train)
    X_valid = replacer.transform(X_valid)

    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    X_train_encoded = encoder.fit_transform(X_train[cat_cols])
    X_valid_encoded = encoder.transform(X_valid[cat_cols])

    # Create a DataFrame with the encoded columns
    X_train_encoded = pd.DataFrame(X_train_encoded, columns=encoder.get_feature_names_out(cat_cols))
    X_valid_encoded = pd.DataFrame(X_valid_encoded, columns=encoder.get_feature_names_out(cat_cols))

    # Drop the original categorical columns from the training and validation sets
    X_train = X_train.drop(cat_cols, axis=1)
    X_valid = X_valid.drop(cat_cols, axis=1)

    # Concatenate the numerical and encoded categorical columns
    X_train = pd.concat([X_train, X_train_encoded], axis=1)
    X_valid = pd.concat([X_valid, X_valid_encoded], axis=1)

    # Scale the numerical columns using the StandardScaler
    scaler = StandardScaler()
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_valid[num_cols] = scaler.transform(X_valid[num_cols])

    # Start the timer
    start_time = time.time()

    # Train a Random Forest model
    model = RandomForestClassifier(n_jobs=-1)

    param_grid = {
    'n_estimators': [550, 600, 650],
    'max_depth': [55, 60, 65],
    'min_samples_split': [4, 5, 6],
    'min_samples_leaf': [1, 2],
    'max_features': ['log2'],
    'bootstrap': [True],
    'criterion': ['entropy'],
    'max_samples': [0.85, 0.9, 0.95]
    }

    # Use make_scorer to create a scorer for the Matthews correlation coefficient
    mcc_scorer = make_scorer(matthews_corrcoef)

    # Initialize GridSearchCV
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring=mcc_scorer,  # Using MCC as the metric for scoring
        cv=5,
        verbose=2,
        n_jobs=-1
    )


    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_

    y_pred = best_model.predict(X_valid)

    # Calculate the Matthews correlation coefficient
    mcc = matthews_corrcoef(y_valid, y_pred)
    print(f"Fold={fold}, MCC={mcc}")
    print(f"Best parameters found: {grid_search.best_params_}")
    print(f"Best MCC score: {grid_search.best_score_}")

    # End the timer
    end_time = time.time()

    # Calculate the elapsed time
    elapsed_time = end_time - start_time

    print(f"Model building took {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    # for fold_ in range(5):
    #     run(fold_)
    run(0)

