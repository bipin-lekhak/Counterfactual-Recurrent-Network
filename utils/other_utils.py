import logging
import json
import pickle

import pandas as pd

from pathlib import Path


def write_results_to_file_pickle(filename, data):
    with open(filename, "wb") as handle:
        pickle.dump(data, handle, protocol=2)


def append_results_to_file_pickle(filename, data):
    with open(filename, "a+b") as handle:
        pickle.dump(data, handle, protocol=2)


def write_results_to_file_json(filename, data):
    with open(filename, "w") as handle:
        json.dump(data, handle)


def append_results_to_file_json(filename, data):
    with open(filename, "a+") as handle:
        json.dump(data, handle)


def read_results_from_file_json(filename):
    with open(filename, "r") as handle:
        data = json.load(handle)
    return data


def get_hyperparams(model, radio_coef, chemo_coef):
    if model not in {"encoder", "decoder"}:
        logging.error("Invalid model type")
        return None
    if radio_coef != chemo_coef:
        logging.info("Coefs not found")
        return None

    hp_df = pd.read_csv(Path(__file__).parent/"hparams.csv")
    hp_filt = hp_df[
        (hp_df["model"] == model)
        & (hp_df["radio_coef"] == radio_coef)
        & (hp_df["chemo_coef"] == chemo_coef)
    ]
    if len(hp_filt) == 0:
        logging.info("No hyperparameters found")
        return None
    if len(hp_filt) > 1:
        logging.info("Multiple hyperparameters found")
        return None
    best_hyperparams = hp_filt.iloc[0, 3:].to_dict()
    return best_hyperparams

