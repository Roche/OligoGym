import pandas as pd
from fuzzywuzzy import process

from .data import Dataset

import importlib.resources as pkg_resources
from oligogym.resources import pkg_dataset
import json


class DatasetDownloader:
    """
    A class used to download datasets.

    Attributes:
        all_datasets_info (pandas.DataFrame): Information about all available datasets.
    """

    def __init__(self):
        """
        Constructs all the necessary attributes for the DatasetDownloader object.

        """
        with pkg_resources.open_binary(pkg_dataset, "all_datasets_info.csv.gz") as file:
            self.all_datasets_info = pd.read_csv(file, compression='gzip')

    def download(self, dataset_key, verbose=0):
        """
        Downloads a dataset or all datasets from Posit Connect.

        Args:
            dataset_key (str): The key of the dataset to download. If 'all', all datasets are downloaded.
            verbose (int, optional): The verbosity level. Defaults to 0.

        Returns:
            list or Dataset: The downloaded datasets. If 'all' was passed, a list of all datasets is returned.
                             Otherwise, the single downloaded dataset is returned.
        """
        datasets = []
        if dataset_key == "all":
            for dataset_name in self.all_datasets_info["name"]:
                if verbose > 0:
                    print(f"Downloading {dataset_name}")

                with pkg_resources.open_binary(pkg_dataset, f"{dataset_name}_processed.csv.gz") as file:
                    df = pd.read_csv(file, compression='gzip')

                with pkg_resources.open_binary(pkg_dataset, f"{dataset_name}_processed.json") as file:
                    json_dict = json.load(file)

                dataset = Dataset(json_dict)
                dataset.build(df)
                datasets.append(dataset)
            if verbose > 0:
                print(f"All datasets have been successfully downloaded.")
            return tuple(datasets)
        else:
            closest_match_key, match_score_key = process.extractOne(
                dataset_key, self.all_datasets_info["key"].to_list()
            )
            closest_match_name, match_score_name = process.extractOne(
                dataset_key, self.all_datasets_info["name"].to_list()
            )

            if match_score_key > match_score_name:
                index_of_closest_match = self.all_datasets_info[
                    self.all_datasets_info["key"] == closest_match_key
                ].index[0]
                closest_match = self.all_datasets_info.iloc[index_of_closest_match][
                    "name"
                ]
                match_score = match_score_key
            else:
                closest_match = closest_match_name
                match_score = match_score_name

            if match_score < 50:
                print(f"Error: The provided dataset_key '{dataset_key}' is not valid.")
                return None
            else:
                with pkg_resources.open_binary(pkg_dataset, f"{closest_match}_processed.csv.gz") as file:
                    df = pd.read_csv(file, compression='gzip')

                with pkg_resources.open_binary(pkg_dataset, f"{closest_match}_processed.json") as file:
                    json_dict = json.load(file)

                dataset = Dataset(json_dict)
                dataset.build(df)
                if verbose > 0:
                    print(
                        f"Dataset '{closest_match}' has been successfully downloaded."
                    )
                return dataset

    def show_available_datasets(self, full_info=False):
        """
        Prints the available datasets in Posit Connect.

        Args:
            full_info (bool, optional): Whether to print full information about the datasets. Defaults to False.
        """
        if full_info:
            print(self.all_datasets_info)
        else:
            print(self.all_datasets_info["name"])
