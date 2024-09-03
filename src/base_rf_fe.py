# This script trains a Random Forest model on the training data with 5-fold cross-validation
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from clean import RareCategoryReplacer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import matthews_corrcoef, accuracy_score
from analyze import missing_values
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import matthews_corrcoef


def run(fold):
    # Load the training data with folds
    df = pd.read_csv("./input/train_folds.csv")
    df = df.sample(frac=0.1, random_state=0)

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

    cat_cols =X_train.select_dtypes(include='object').columns

    replacer = RareCategoryReplacer(columns=cat_cols, proportion_threshold=0.01)
    X_train = replacer.fit_transform(X_train)
    X_valid = replacer.transform(X_valid)

    # Feature Engineering
    X_train['cap-surface_cap-shape'] = X_train['cap-surface'] + '_' + X_train['cap-shape']
    X_train['gill-attachment_gill-color'] = X_train['gill-attachment'] + '_' + X_train['gill-color']
    X_train['gill-spacing_gill-color'] = X_train['gill-spacing'] + '_' + X_train['gill-color']
    X_train['gill-color_veil-color'] = X_train['gill-color'] + '_' + X_train['veil-color']
    X_train['stem-root_stem-color'] = X_train['stem-root'] + '_' + X_train['stem-color']
    X_train['stem-surface_gill-color'] = X_train['stem-surface'] + '_' + X_train['gill-color']
    X_train['veil-type_cap-shape'] = X_train['veil-type'] + '_' + X_train['cap-shape']
    X_train['veil-color_gill-color'] = X_train['veil-color'] + '_' + X_train['gill-color']
    X_train['spore-print-color_gill-color'] = X_train['spore-print-color'] + '_' + X_train['gill-color']


    X_valid['cap-surface_cap-shape'] = X_valid['cap-surface'] + '_' + X_valid['cap-shape']
    X_valid['gill-attachment_gill-color'] = X_valid['gill-attachment'] + '_' + X_valid['gill-color']
    X_valid['gill-spacing_gill-color'] = X_valid['gill-spacing'] + '_' + X_valid['gill-color']
    X_valid['gill-color_veil-color'] = X_valid['gill-color'] + '_' + X_valid['veil-color']
    X_valid['stem-root_stem-color'] = X_valid['stem-root'] + '_' + X_valid['stem-color']
    X_valid['stem-surface_gill-color'] = X_valid['stem-surface'] + '_' + X_valid['gill-color']
    X_valid['veil-type_cap-shape'] = X_valid['veil-type'] + '_' + X_valid['cap-shape']
    X_valid['veil-color_gill-color'] = X_valid['veil-color'] + '_' + X_valid['gill-color']
    X_valid['spore-print-color_gill-color'] = X_valid['spore-print-color'] + '_' + X_valid['gill-color']

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

    # Train a Random Forest model
    model = RandomForestClassifier(n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_valid)

    # Calculate the Matthews correlation coefficient
    mcc = matthews_corrcoef(y_valid, y_pred)
    print(f"Fold={fold}, MCC={mcc}")

    # Calculate the accuracy
    print(f"Fold={fold}, Accuracy={accuracy_score(y_valid, y_pred)}")


if __name__ == "__main__":
    for fold_ in range(5):
        run(fold_)
    # run(0)

