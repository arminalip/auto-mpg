import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from time import time

import numpy as np
from sklearn.model_selection import GridSearchCV
import pickle

import sys
sys.path.insert(0,'/Users/Armin/Desktop/MyFirstMLOps/auto-mpg/')

import src
from src.clean import load_raw_data

# Continous features
CONTINUOUS_FEATURES = ["displacement", "horsepower", "weight", "acceleration"]
# Categorical features
ORDINAL_FEATURES = ["cylinders", "year"]
NOMINAL_FEATURES = ["region"]


def make_final_transformation_pipe():

    # Build transformation pipelines adapted to feature types
    cont_pipeline = Pipeline(
        [
            ("imputer_cont", SimpleImputer(strategy="median")),
            ("std_scaler_cont", StandardScaler()),
        ]
    )

    ord_pipeline = Pipeline(
        [
            ("imputer_ord", SimpleImputer(strategy="most_frequent")),
            ("std_scaler_ord", StandardScaler()),
        ]
    )

    full_pipeline = ColumnTransformer(
        [
            ("cont", cont_pipeline, CONTINUOUS_FEATURES),
            ("ord", ord_pipeline, ORDINAL_FEATURES),
            ("nom", OneHotEncoder(), NOMINAL_FEATURES),
        ]
    )

    return full_pipeline


def get_cleaned_train_test_df():
    clean_data_path = src.utils.data_path("interim", "data_cleaned.pkl")
    df = pd.read_pickle(clean_data_path)
    X = df.drop("mpg", axis=1)
    y = df["mpg"]
    
    return train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)


def make_final_sets():
    X_train, X_test, y_train, y_test = get_cleaned_train_test_df()

    full_pipeline = make_final_transformation_pipe()
    X_train_processed_values = full_pipeline.fit_transform(X_train)
    X_test_processed_values = full_pipeline.transform(X_test)

    # Add column names to build the processed dataframe
    region_ohe_features = list(
        full_pipeline.named_transformers_["nom"].get_feature_names()
    )
    column_names = CONTINUOUS_FEATURES + ORDINAL_FEATURES + region_ohe_features
    X_train_processed = pd.DataFrame(X_train_processed_values, columns=column_names)
    X_test_processed = pd.DataFrame(X_test_processed_values, columns=column_names)

    # Drop one of the ohe features to limit correlations in the data set
    for df in (X_train_processed, X_test_processed):
        df.drop("x0_EUROPE", axis=1, inplace=True)

    # Save the data
    df_train_processed = X_train_processed.join(y_train.reset_index(drop=True))
    src.utils.save_data(df_train_processed, "processed", "train_processed.pkl")

    df_test_processsed = X_test_processed.join(y_test.reset_index(drop=True))
    src.utils.save_data(df_test_processsed, "processed", "test_processed.pkl")

    return df_train_processed, df_test_processsed

def RF(df_train):

    X_train = df_train.drop("mpg", axis=1)
    y_train = df_train["mpg"]

    rf = RandomForestRegressor(
        max_depth=5,
        n_estimators=100,
        bootstrap=True,
        max_features=3,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    return rf

def save_model(rf):
    ##dump the model into a file
    with open("model.bin", 'wb') as f_out:
        pickle.dump(rf, f_out) # write final_model in .bin file
        f_out.close()  # close the file 

def predict_mpg(config, model):
    
    if type(config) == dict:
        df = pd.DataFrame(config)
    else:
        df = config

    y_pred = model.predict(df)
    return y_pred

df = load_raw_data()
df["horsepower"] = pd.to_numeric(df["horsepower"], errors="coerce")
src.clean.correct_company_names(df)
src.clean.get_region_names(df)
src.clean.save_data(df, "interim", "data_cleaned.pkl")

df_train, df_test = make_final_sets()
rf = RF(df_train)
save_model(rf)
