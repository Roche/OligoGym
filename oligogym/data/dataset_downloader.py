import pandas as pd

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
            all_keys = [i.lower() for i in self.all_datasets_info["key"].to_list()]

            dataset_map = dict(zip(all_keys, self.all_datasets_info["name"].to_list()))

            # Explicitly raise an exception if the dataset_key is not valid
            if dataset_key.lower() not in all_keys:
                raise ValueError(f"Error: The provided dataset_key '{dataset_key}' is not valid.")

            print(dataset_map[dataset_key.lower()])
            with pkg_resources.open_binary(pkg_dataset, f"{dataset_map[dataset_key.lower()]}_processed.csv.gz") as file:
                df = pd.read_csv(file, compression='gzip')

            with pkg_resources.open_binary(pkg_dataset, f"{dataset_map[dataset_key.lower()]}_processed.json") as file:
                json_dict = json.load(file)

            dataset = Dataset(json_dict)
            dataset.build(df)
            if verbose > 0:
                print(
                    f"Dataset '{dataset_map[dataset_key.lower()]}' has been successfully downloaded."
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