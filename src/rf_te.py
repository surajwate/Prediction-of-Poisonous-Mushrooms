# Random Forest with Target Encoding
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
from category_encoders import TargetEncoder


def run(fold):
    # Load the training data with folds
    df = pd.read_csv("./input/train_folds.csv")

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

    # Target encoding
    # Map the target values to numbers
    target_mapping = {"p": 0, "e": 1}
    y_train_num = y_train.map(target_mapping)

    # Create an instance of the TargetEncoder
    encoder = TargetEncoder(cols=cat_cols)

    # Fit the encoder on the training data and transform the training and validation data
    X_train = encoder.fit_transform(X_train, y_train_num)
    X_valid = encoder.transform(X_valid)

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