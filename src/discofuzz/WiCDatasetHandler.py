from typing import Tuple
import os

import wget as wget
import zipfile
import pandas as pd

class WiCDatasetHandler:
    @staticmethod
    def _download_dataset():
        wget.download("https://pilehvar.github.io/wic/package/WiC_dataset.zip")

        # Define the path to your zip file and the target directory for extraction
        zip_file_path = 'WiC_dataset.zip'
        extract_dir = 'WiC_dataset'

        # Create the target directory if it doesn't exist
        os.makedirs(extract_dir, exist_ok=True)

        # Open the zip file in read mode ('r')
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            # Extract all contents to the specified directory
            zip_ref.extractall(extract_dir)

        print(f"Contents of '{zip_file_path}' extracted to '{extract_dir}'")
    

    @staticmethod
    def load_dataset(path:str) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        # Try loading training data
        try:
            X_train = pd.read_csv(f"{path}/WiC_dataset/train/train.data.txt", sep="\t", header=None)
        # if unable to load dataset, try downloading it again
        except Exception as e:
            print(e)
            WiCDatasetHandler._download_dataset()
            X_train = pd.read_csv(f"{path}/WiC_dataset/train/train.data.txt", sep="\t", header=None)
        
        X_test = pd.read_csv(f"{path}/WiC_dataset/test/test.data.txt", sep="\t", header=None)
        y_train = pd.read_csv(f"{path}/WiC_dataset/train/train.gold.txt", header=None)
        y_test = pd.read_csv(f"{path}/WiC_dataset/test/test.gold.txt", header=None)
        
        X_train.columns = ["lemma", "pos", "index1-index2", "sent_1", "sent_2"]
        X_test.columns = ["lemma", "pos", "index1-index2", "sent_1", "sent_2"]

        # Load ground truth labels
        y_train = y_train[0].apply(lambda x: 1 if x == "T" else 0)
        y_test = y_test[0].apply(lambda x: 1 if x == "T" else 0)

        # clean training dataframe
        X_train["pos"] = X_train["pos"].apply(lambda x: x.lower())
        X_train["tok_idx_1"] = X_train["index1-index2"].apply(lambda x: int(x.split("-")[0]))
        X_train["tok_idx_2"] = X_train["index1-index2"].apply(lambda x: int(x.split("-")[1]))
        X_train.drop("index1-index2", axis=1, inplace=True)

        # clean testing dataframe
        X_test["pos"] = X_test["pos"].apply(lambda x: x.lower())
        X_test["tok_idx_1"] = X_test["index1-index2"].apply(lambda x: int(x.split("-")[0]))
        X_test["tok_idx_2"] = X_test["index1-index2"].apply(lambda x: int(x.split("-")[1]))
        X_test.drop("index1-index2", axis=1, inplace=True)

        return X_train, y_train, X_test, y_test