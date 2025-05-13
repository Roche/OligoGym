import json
import os

import numpy as np
import pandas as pd
import scipy

from ..stats import *
from ..split import (
    random_split,
    target_split,
    backbone_split,
    nucleobase_split,
    time_split,
)


class Dataset:
    """
    A custom dataset object that is defined by a JSON file or a dictionary.

    This class allows for the creation of a dataset object with attributes set based on the key-value pairs in the JSON file or dictionary. The dataset object also includes methods for building the dataset from 'x', 'y', and 'targets' inputs, retrieving these inputs, and applying statistical functions to each sequence in 'x'.

    Attributes:
        data (pandas.DataFrame): A DataFrame that holds the 'x', 'y', and 'targets' columns.
    """

    def __init__(self, json_content=None):
        """
        Initializes the dataset object by loading attributes from a JSON file or a dictionary.

        Args:
            json_content (str or dict, optional): The path to the JSON file or a dictionary. Defaults to None.

        Raises:
            ValueError: If the JSON file does not exist.
            TypeError: If json_content is not a path to a JSON file or a dictionary.
        """
        if json_content is None:
            attributes = {}
        elif isinstance(json_content, str):
            if os.path.isfile(json_content):
                with open(json_content, "r") as f:
                    attributes = json.load(f)
            else:
                raise ValueError(f"No such file: '{json_content}'")
        elif isinstance(json_content, dict):
            attributes = json_content
        else:
            raise TypeError(
                "json_content must be a path to a JSON file or a dictionary"
            )

        for key, value in attributes.items():
            setattr(self, key, value)
        self.data = None

    def build(self, x=None, y=None, targets=None, dataframe=None):
        """
        Builds the data attribute from x, y, and targets inputs or a DataFrame.

        Args:
            x (numpy.ndarray, optional): The 'x' column.
            y (numpy.ndarray, optional): The 'y' column.
            targets (numpy.ndarray, optional): The 'targets' column.
            dataframe (pandas.DataFrame, optional): A DataFrame containing 'x', 'y', and 'targets' columns.

        Raises:
            ValueError: If the DataFrame does not contain 'x', 'y', and 'targets' columns.
            ValueError: If x, y, and targets are not provided when dataframe is not provided.
        """
        inputs = [x, y, targets, dataframe]
        non_none_inputs = [inp for inp in inputs if inp is not None]

        if len(non_none_inputs) == 1:
            dataframe = non_none_inputs[0]

        if dataframe is not None:
            if not all(col in dataframe.columns for col in ["x", "y", "targets"]):
                raise ValueError(
                    "DataFrame must contain 'x', 'y', and 'targets' columns"
                )
            self.data = dataframe
        else:
            if x is None or y is None or targets is None:
                raise ValueError(
                    "x, y, and targets must be provided if dataframe is not provided"
                )
            self.data = pd.DataFrame({"x": x, "y": y, "targets": targets})

        self._create_properties()

    def _create_properties(self):
        """
        Dynamically creates properties for each column in the data DataFrame.

        This method iterates over each column in the `data` DataFrame and creates a property for it. The property allows access to the column values directly as an attribute of the class instance.
        """
        for column in self.data.columns:
            setattr(
                self.__class__,
                column,
                property(lambda self, col=column: self.data[col].values),
            )

    def get_helm_stats(self, format="aggregate", cosine_dist=False):
        """
        Applies statistics function to each helm sequence in 'x'.

        If format is 'individual', returns the results for each helm sequence.
        If format is 'aggregate', returns the average and combined stats.

        Args:
            format (str, optional): The format of the results. Defaults to 'aggregate'.
            cosine_dist (bool, optional): Whether to calculate cosine distance to nearest neighbour or not. Defaults to False.

        Returns:
            pandas.DataFrame: A DataFrame with the results.
        """
        stats = []
        for helm in self.x:
            stats_tmp = {}
            strands = get_strands(helm)
            for strand in strands:
                nt_seq_len = get_nt_seq_len(helm, strand=strand)
                unique_monomers = get_unique_monomers(helm, strand=strand)
                gc_content = get_gc_content(helm, strand=strand)
                g_content, c_content, a_content, tu_content = get_nt_content(
                    helm, strand=strand
                )
                xna_base, xna_sugar, xna_phosphate = get_xna(helm, strand=strand)
                stats_tmp.update(
                    {
                        f"{strand}_nt_seq_len": nt_seq_len,
                        f"{strand}_unique_monomers": unique_monomers,
                        f"{strand}_GC_content": gc_content,
                        f"{strand}_G_content": g_content,
                        f"{strand}_C_content": c_content,
                        f"{strand}_A_content": a_content,
                        f"{strand}_TU_content": tu_content,
                        f"{strand}_xna_base": xna_base,
                        f"{strand}_xna_sugar": xna_sugar,
                        f"{strand}_xna_phosphate": xna_phosphate,
                    }
                )
            stats.append(stats_tmp)

        df = pd.DataFrame(stats)

        df["uniqueness"] = get_uniqueness(self.x)

        if cosine_dist:
            df["cosine_dist_to_nn"] = get_cosine_dist_to_nearest_neighbor(self.x)

        if format == "individual":
            return df
        elif format == "aggregate":
            stats = pd.DataFrame(
                [
                    {
                        "avg_nt_seq_len": np.mean(df.filter(like="nt_seq_len")),
                        "combined_unique_monomers": sorted(
                            list(
                                set(
                                    monomer
                                    for monomers in df.filter(
                                        like="unique_monomers"
                                    ).stack()
                                    for monomer in monomers
                                )
                            )
                        ),
                        "avg_GC_content": np.mean(df.filter(like="GC_content")),
                        "avg_G_content": np.mean(df.filter(like="G_content")),
                        "avg_C_content": np.mean(df.filter(like="C_content")),
                        "avg_A_content": np.mean(df.filter(like="A_content")),
                        "avg_TU_content": np.mean(df.filter(like="TU_content")),
                        "num_duplicates": (df["uniqueness"] == 0).sum(),
                    }
                ]
            )
            if cosine_dist:
                stats["avg_cosine_dist"] = df["cosine_dist_to_nn"].mean()
            return stats

    def get_label_stats(self, task=None):
        """
        Calculates and returns statistics about the labels based on the task type.

        Args:
            task (str, optional): The type of task. Can be "regression", "binary classification", or "multiclass classification".
                                  If not provided, it will use self.task.

        Returns:
            pandas.DataFrame: A DataFrame containing label statistics based on the task type.
                For regression task, it includes the statistics returned by scipy.stats.describe and the number of zero values.
                For binary classification task, it includes the number of observations and the percentage of positive labels.
                For multiclass classification task, it includes the number of observations, the list of classes,
                the number of classes, the modal class, and the number of observations per class.

        Raises:
            ValueError: If neither task nor self.task is provided, or if the task is not one of the allowed values.
        """
        if task is None:
            if hasattr(self, "task"):
                task = self.task
            else:
                raise ValueError(
                    "Task must be provided either as an argument or as an attribute of the object."
                )

        task = task.lower()
        if task not in [
            "regression",
            "binary classification",
            "multiclass classification",
        ]:
            raise ValueError(
                "Task must be 'regression', 'binary classification', or 'multiclass classification'."
            )

        if task == "regression":
            label_stats = scipy.stats.describe(self.y)
            num_zeros = np.sum(self.y == 0)
            label_stats = pd.DataFrame([label_stats._asdict()])
            label_stats["num_zeros"] = num_zeros
            return label_stats
        elif task == "binary classification":
            nobs = len(self.y)
            percent_pos = np.sum(self.y == 1) / len(self.y)
            return pd.DataFrame([{"nobs": nobs, "percent_pos": percent_pos}])
        elif task == "multiclass classification":
            nobs = len(self.y)
            classes = list(set(self.y))
            num_classes = len(classes)
            num_obs_per_class = [np.sum(self.y == i) for i in classes]
            modal_class = classes[np.argmax(num_obs_per_class)]
            num_obs_per_class = pd.DataFrame([dict(zip(classes, num_obs_per_class))])

            label_stats = pd.DataFrame(
                [
                    {
                        "nobs": nobs,
                        "classes": classes,
                        "num_classes": num_classes,
                        "modal_class": modal_class,
                    }
                ]
            )
            label_stats = pd.concat([label_stats, num_obs_per_class], axis=1)
            return label_stats

    def split(
        self,
        split_strategy=None,
        test_size=0.2,
        val_size=None,
        random_state=None,
        return_index=False,
        kmer_max=3,
        kmer=True,
        cutoff_date=None,
        targets=None,
        timestamp=None,
        strand="RNA1",
        **kwargs,
    ):
        """
        Performs train/test/validation split on the dataset based on the split strategy.

        Args:
            split_strategy (str, optional): The strategy to use for splitting the dataset.
                Options are 'random', 'stratified', 'target', 'backbone', 'nucleobase', and 'time'.
                Defaults to the instance's `rec_split` attribute.
            test_size (float, optional): The proportion of the dataset to include in the test split. Defaults to 0.2.
            val_size (float, optional): The proportion of the dataset to include in the validation split. Defaults to None.
            random_state (int, optional): The seed used by the random number generator. Defaults to None.
            return_index (bool, optional): Whether to return the indices of the train/test split. Defaults to False.
            kmer_max (int, optional): The maximum length of the kmer to use for creating the kmer frequency matrix if nucleobase splitting strategy is used. Defaults to 3.
            kmer (bool, optional): Whether to use kmer frequency or fasta identity as label for splitting. Defaults to True.
            cutoff_date (np.datetime64 or list of np.datetime64, optional): The cutoff date(s) for splitting the data. Defaults to None.
            targets (array-like, optional): Targets to use for splitting. Overrides self.targets if provided. Defaults to None.
            timestamp (array-like, optional): Timestamps to use for splitting. Overrides self.timestamp if provided. Defaults to None.
            strand (str, optional): Specify which strands to use for nucleobase and backbone split.
            **kwargs: Additional keyword arguments to pass to the KMeans constructor for nucleobase split.

        Returns:
            tuple: The train/validation/test split of the dataset, and optionally the indices of the split.

        Raises:
            ValueError: If an invalid split strategy is provided or required data is missing.
        """
        if split_strategy:
            self.split_strategy = split_strategy.lower()
        else:
            self.split_strategy = getattr(self, "rec_split", "random").lower()

        x = self.x
        y = self.y

        if targets is not None:
            targets = pd.Series(targets)
        else:
            targets = pd.Series(self.targets) if hasattr(self, "targets") else None

        if timestamp is not None:
            timestamp = pd.Series(timestamp)
        else:
            timestamp = (
                pd.Series(self.timestamp) if hasattr(self, "timestamp") else None
            )

        if self.split_strategy == "target" and targets is None:
            raise ValueError("Targets must be provided for 'target' split strategy.")
        if self.split_strategy == "time" and timestamp is None:
            raise ValueError("Timestamps must be provided for 'time' split strategy.")

        if self.split_strategy == "random":
            return random_split(x, y, test_size, val_size, random_state, return_index)
        elif self.split_strategy == "stratified":
            return random_split(
                x, y, test_size, val_size, random_state, return_index, stratified=True
            )
        elif self.split_strategy == "target":
            return target_split(
                x, y, targets, test_size, val_size, random_state, return_index
            )
        elif self.split_strategy == "backbone":
            return backbone_split(
                x, y, test_size, val_size, random_state, return_index, strand
            )
        elif self.split_strategy == "nucleobase":
            return nucleobase_split(
                x,
                y,
                test_size,
                val_size,
                random_state,
                return_index,
                kmer_max,
                kmer,
                strand,
                **kwargs,
            )
        elif self.split_strategy == "time":
            return time_split(
                x,
                y,
                timestamp,
                test_size,
                val_size,
                return_index,
                cutoff_date,
            )
        else:
            raise ValueError(f"Invalid split strategy: '{self.split_strategy}'")
