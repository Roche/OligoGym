import importlib.resources
import os
import pickle
import sys
import warnings
from collections import defaultdict
from itertools import zip_longest, product
from collections import Counter
from fuzzywuzzy import process
from typing import Dict, List, Optional, Union

from Bio import Entrez, SeqIO
from Bio.Seq import Seq

import numpy as np
import pandas as pd
import RNA

from .helm import helm2xna, xna2helm
from .utils import count_overlapping, merge_dicts

ORDERED_COMPONENTS = ["phosphate", "sugar", "base"]
DG_RNA = {
    "AA": -0.93,
    "UU": -0.93,
    "AU": -1.10,
    "UA": -1.33,
    "CU": -2.08,
    "AG": -2.08,
    "CA": -2.11,
    "UG": -2.11,
    "GU": -2.24,
    "AC": -2.24,
    "GA": -2.35,
    "UC": -2.35,
    "CG": -2.36,
    "GG": -3.26,
    "CC": -3.26,
    "GC": -3.42,
    "init": 4.09,
    "endAU": 0.45,
    "sym": 0.43,
}
DG_HYBRID = {
    "TT": -0.7,
    "GT": -1.5,
    "CT": -1.3,
    "AT": -0.4,
    "TG": -1.2,
    "GG": -1.7,
    "CG": -1.4,
    "AG": -0.4,
    "TC": -1.5,
    "GC": -2.0,
    "CC": -2.3,
    "AC": -1.4,
    "TA": -0.5,
    "GA": -1.4,
    "CA": -1.6,
    "AA": 0.2,
    "init_CG": 2.0,
    "init_TA": 2.6,
}


def _extract_monomers(
    oligo_helm: str, strands: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Extracts monomers from an oligonucleotide HELM notation string.

    This function reads an oligonucleotide HELM notation string, converts it to XNA format, and extracts the monomers (sugar,
    base, and phosphate) for RNA polymers. The extracted monomers are returned as
    a pandas DataFrame.

    Args:
        oligo_helm (str): The oligonucleotide HELM notation string.

    Returns:
        pd.DataFrame: A DataFrame containing the extracted monomers with columns
        "sugar", "base", and "phosphate". Missing values are filled with "NA".
    """
    xna = helm2xna(oligo_helm)
    polymers = pd.Series(xna.polymers.keys())
    polymers = polymers[polymers.str.contains("RNA|CHEM")]

    if strands is not None:
        assert all(strand in polymers.values for strand in strands), "Invalid strands"
        polymers = list(set(polymers).intersection(set(strands)))

    monomer_dfs = []
    for key in polymers:
        if "RNA" in key:
            sugar = xna.polymers[key]["sugar"].split(".")
            base = xna.polymers[key]["base"].split(".")
            phosphate = xna.polymers[key]["phosphate"].split(".")
            monomer_dfs.append(
                pd.DataFrame(
                    zip_longest(sugar, base, phosphate),
                    columns=["sugar", "base", "phosphate"],
                ).assign(polymer=key)
            )
    monomers = pd.concat(monomer_dfs)
    monomers.replace("", "EMPTY", inplace=True)
    monomers.fillna("NA", inplace=True)
    return monomers


class OneHotEncoder:
    """
    OneHotEncoder is a class for encoding oligo sequences into one-hot encoded features.

    This class provides methods to fit, transform, and reverse transform oligo sequences
    using one-hot encoding. It supports encoding components such as phosphate, sugar, and base.

    Attributes:
        encode_components (List[str]): List of components to encode.
        encoding_maps (Dict[str, Dict[str, list]]): Mapping of components to one-hot encoding.
        max_sequence_length (int): Maximum length of the oligo sequences.

        Args:
            encode_components (List[str], optional): A list of components to encode.
                Must be a subset of ["phosphate", "sugar", "base"]. Defaults to ["phosphate", "sugar", "base"].
            encoding_maps (Optional[Dict[str, Dict[str, list]]], optional): A dictionary containing encoding maps for each component.
                Defaults to an empty dictionary.
            strands (Optional[List[str]], optional): List of strands to consider. Defaults to None, meaning all strands will be featurized.
            max_sequence_length (Optional[int], optional): Maximum length of the oligo sequences. Defaults to None, in which case the max
                sequence length will be inferred from the data. If the provided max_sequence_length is shorter than the one inferred from the
                data, it will be overriden. This parameter is useful for padding sequences to a fixed length bigger than the one seen in the
                training data.
        Raises:
            AssertionError: If any component in `encode_components` is not in ["sugar", "base", "phosphate"].
    """

    def __init__(
        self,
        encode_components: List[str] = ["phosphate", "sugar", "base"],
        encoding_maps: Optional[Dict[str, Dict[str, list]]] = dict(),
        strands: Optional[List[str]] = None,
        max_sequence_length: Optional[int] = None,
    ):
        """
        Initializes the feature encoder.
        """
        assert all(
            component in ["sugar", "base", "phosphate"]
            for component in encode_components
        ), "Invalid encode components"
        self.encode_components = [
            item for item in ORDERED_COMPONENTS if item in encode_components
        ]  # reorder components to fixed order
        self.encoding_maps = encoding_maps.copy()
        self.strands = strands
        self.max_sequence_length = (
            max_sequence_length if max_sequence_length is not None else -1
        )

    def _extract_monomers(self, oligo_helm: str) -> pd.DataFrame:
        """
        Extracts monomers from the given oligo HELM string.

        Args:
            oligo_helm (str): The HELM notation string representing the oligonucleotide.

        Returns:
            pd.DataFrame: A DataFrame containing the extracted monomers.
        """
        return _extract_monomers(oligo_helm, self.strands)

    def _encode_monomers(self, monomers: pd.DataFrame) -> pd.DataFrame:
        """
        Encode monomers using the specified encoding maps.

        Args:
            monomers (pd.DataFrame): DataFrame containing the monomers to encode.

        Returns:
            pd.DataFrame: DataFrame containing the encoded monomers.
        """
        encoded_monomers = monomers[self.encode_components].copy()
        for component in self.encode_components:
            encoded_monomers[component] = monomers[component].apply(
                lambda x: self.encoding_maps[component].get(
                    x, [0] * len(self.encoding_maps[component])
                )
            )
        return encoded_monomers

    def _get_dictionaries(self, oligo_list: list) -> None:
        """
        Build encoding dictionaries based on the provided oligo sequences.

        Args:
            oligo_list (list): List of oligo sequences.
        """
        monomers = [self._extract_monomers(oligo) for oligo in oligo_list]
        monomers = pd.concat(monomers)

        for component in self.encode_components:
            if component not in self.encoding_maps.keys():
                unique_elements = monomers[component].unique()
                self.encoding_maps[component] = {
                    element: OneHotEncoder.one_hot_encode(unique_elements, element)
                    for element in unique_elements
                }

    def _extract_features(self, oligo_helm: str) -> np.ndarray:
        """
        Extract features from a single oligo sequence.

        Args:
            oligo_helm (str): Oligonucleotide in HELM notation.

        Returns:
            np.ndarray: Array containing the extracted one-hot-encoded features.
        """
        monomers = self._extract_monomers(oligo_helm)
        encoded_monomers = self._encode_monomers(monomers)
        features = np.stack(encoded_monomers.sum(axis=1))
        return features

    def fit_transform(self, oligo_list: list, flatten: bool = False) -> np.ndarray:
        """
        Fit the OneHotEncoder and transform a list of oligo sequences in HELM notation.
        This method first builds encoding dictionaries based on the provided oligo sequences,
        then extracts features from each oligo sequence and encodes them using the dictionaries.

        Args:
            oligo_list (list): List of oligo sequences in HELM notation.
            flatten (bool, optional): Whether to flatten the array. Defaults to False.

        Returns:
            np.ndarray: Array containing the one-hot encoded features.
        """
        self._get_dictionaries(oligo_list)
        features = []

        max_sequence_length = -1
        for oligo in oligo_list:
            extracted_features = self._extract_features(oligo)
            if len(extracted_features) > max_sequence_length:
                max_sequence_length = len(extracted_features)
            features.append(extracted_features)

        if self.max_sequence_length < max_sequence_length:
            self.max_sequence_length = max_sequence_length
        # TODO: strand-wise padding
        for i, feature in enumerate(features):
            if len(feature) < self.max_sequence_length:
                padding = np.zeros(
                    (self.max_sequence_length - len(feature), feature.shape[1])
                )
                features[i] = np.concatenate([feature, padding])
        features = np.stack(features)
        if flatten:
            features = features.reshape(features.shape[0], -1)
        return features

    def transform(self, oligo_list: list, flatten: bool = False) -> np.ndarray:
        """
        Transform a list of oligo sequences into one-hot encoded features. This method does
        not build new encoding dictionaries, but uses the existing ones.

        Args:
            oligo_list (list): List of oligo sequences.
            flatten (bool, optional): Whether to flatten the array. Defaults to False.

        Returns:
            np.ndarray: Array containing the one-hot encoded features.
        """
        assert self.encoding_maps, "OneHotEncoder not fitted"

        features = []

        for oligo in oligo_list:
            extracted_features = self._extract_features(oligo)
            if len(extracted_features) < self.max_sequence_length:
                padding = np.zeros(
                    (
                        self.max_sequence_length - len(extracted_features),
                        extracted_features.shape[1],
                    )
                )
                extracted_features = np.concatenate([extracted_features, padding])
            elif len(extracted_features) > self.max_sequence_length:
                warnings.warn(
                    "Oligo sequence length exceeds maximum sequence length, truncating sequences"
                )
                extracted_features = extracted_features[: self.max_sequence_length]
            features.append(extracted_features)
        features = np.stack(features)
        if flatten:
            features = features.reshape(features.shape[0], -1)
        return np.stack(features)

    def reverse_transform(self, features, return_helms=True) -> list:
        """
        Transforms one-hot-encoded features back to XNA or HELM format.

        Args:
            features (np.ndarray): One-hot-encoded features.
            return_helms (bool): Flag for returning HELMs instead of XNAData.

        Returns:
            list: List containing the reverse transformed features.
        """
        assert self.encode_components == [
            "phosphate",
            "sugar",
            "base",
        ], "Reverse transform is valid only for fully encoded molecules"
        if len(features.shape) == 2:
            features = np.expand_dims(features, axis=0)

        vocab_sizes = {
            component: len(monomers)
            for component, monomers in self.encoding_maps.items()
        }
        self.inverse_maps = {
            component: {
                tuple(monomer): encoding for monomer, encoding in monomers.items()
            }
            for component, monomers in self.encoding_maps.items()
        }
        self.component_features = {}
        start_idx = 0
        for component in vocab_sizes.keys():
            self.component_features[component] = features[
                :, :, start_idx : start_idx + vocab_sizes[component]
            ]
            start_idx += vocab_sizes[component]

        xnas = []
        for i in range(features.shape[0]):
            monomers = defaultdict(list)
            for j in range(features.shape[1]):
                for component in vocab_sizes.keys():
                    monomer_ohe = tuple(
                        self.component_features[component][i, j, :].astype(int)
                    )
                    if any(monomer_ohe):
                        monomers[component].append(
                            self.inverse_maps[component][monomer_ohe]
                        )
            monomers = pd.DataFrame(monomers)
            xna = {
                "RNA1": {
                    "phosphate": ".".join(monomers["phosphate"].values),
                    "sugar": ".".join(monomers["sugar"].values),
                    "base": ".".join(monomers["base"].values),
                }
            }
            if return_helms:
                xnas.append(
                    xna2helm(xna)
                )
            else:
                xnas.append(xna)
        return xnas

    @staticmethod
    def one_hot_encode(unique_values: list, value: str) -> list:
        """
        Perform one-hot encoding for a given value.

        Args:
            unique_values (list): List of unique values.
            value (str): Value to encode.

        Returns:
            list: One-hot encoded representation of the value.
        """
        return [1 if unique_value == value else 0 for unique_value in unique_values]


class KMersCounts:
    """
    A class to count k-mers and modifications in oligonucleotide sequences.

    Args:
        k (Union[list[int], int], optional): The k-mer sizes to be used. If an integer is provided, it will be converted to a list containing that integer. Defaults to 2.
        modification_abundance (bool, optional): Flag to indicate whether to consider modification abundance. Defaults to False.
        split_strands (bool, optional): Flag to indicate whether to split the strands of the oligonucleotide. Defaults to False. Can be overridden when providing a list of strands.
        strands (Optional[List[str]], optional): List of strands to consider. Defaults to None, meaning all strands will be featurized.
    """

    def __init__(
        self,
        k: Union[list[int], int] = 2,
        modification_abundance: bool = False,
        split_strands: bool = False,
        strands: Optional[List[str]] = None,
    ):
        """
        Initializes the KmersCounts featurizers with specified parameters.
        """
        if isinstance(k, int):
            k = [k]
        self.k = k
        self.kmer_entries = []
        self.modification_abundance = modification_abundance
        self.strands = strands
        if (strands is not None) and len(strands) == 1:
            self.split_strands = False
        else:
            self.split_strands = split_strands

    def _extract_monomers(self, oligo_helm: str) -> pd.DataFrame:
        """
        Extract monomers from an oligo sequence.

        Args:
            oligo_helm (str): Oligo sequence.

        Returns:
            pd.DataFrame: DataFrame containing the extracted monomers.
        """
        return _extract_monomers(oligo_helm, self.strands)

    def _extract_kmers(self, fasta_str: Union[str, list[str]]) -> dict:
        """
        Extract k-mers from a given FASTA string.

        Args:
            fasta_str (Union[str, list[str]]): The input FASTA string(s) from which k-mers are to be extracted.

        Returns:
            dict: A dictionary where keys are k-mers and values are their respective counts.
        """
        if not isinstance(fasta_str, list):
            fasta = fasta_str
            kmer_dicts = []
            for k in self.k:
                kmer_counts = defaultdict(int)
                for i in range(len(fasta) - k + 1):
                    kmer = fasta[i : i + k]
                    kmer_counts[f"{kmer}"] += 1
                kmer_dicts.append(kmer_counts)
            kmers_merged = merge_dicts(*kmer_dicts)
        else:
            kmer_dicts = []
            for strand, fasta in enumerate(fasta_str):
                for k in self.k:
                    kmer_counts = defaultdict(int)
                    for i in range(len(fasta) - k + 1):
                        kmer = fasta[i : i + k]
                        kmer_counts[f"RNA{strand+1}_{kmer}"] += 1
                    kmer_dicts.append(kmer_counts)
                kmers_merged = merge_dicts(*kmer_dicts)
        return kmers_merged

    def _count_modifications(self, component_series: pd.Series) -> dict:
        """
        Count the occurrences of each unique value in the given series.

        Args:
            component_series (pd.Series): A pandas Series containing the components to be counted.

        Returns:
            dict: A dictionary where the keys are the unique values from the Series and the values are their respective counts.
        """
        count_dict = component_series.value_counts().to_dict()
        return count_dict

    def _extract_features(self, oligo_helm: str) -> dict:
        """
        Extracts features from an oligonucleotide HELM string.
        This method processes an oligonucleotide HELM string to extract monomers,
        convert them to a FASTA string, and generate k-mers. If modification
        abundance tracking is enabled, it also counts sugar and phosphate
        modifications and merges these counts with the k-mers.

        Args:
            oligo_helm (str): The HELM string representing the oligonucleotide.

        Returns:
            dict: A dictionary containing the k-mers and, if applicable, the
              counts of sugar and phosphate modifications.
        """

        monomers = self._extract_monomers(oligo_helm)
        # Added a quick fix to handle modified bases
        if self.split_strands:
            fasta_str = []
            for strand in monomers["polymer"].unique():
                monomers_strand = monomers.loc[monomers["polymer"] == strand].copy()
                monomers_strand["base"] = monomers_strand["base"].replace("EMPTY", "")
                monomers_strand["base"] = monomers_strand["base"].str[-1]
                fasta_str.append(monomers_strand["base"].str.cat())
            kmers = self._extract_kmers(fasta_str)
        else:
            monomers["base"] = monomers["base"].replace("EMPTY", "")
            monomers["base"] = monomers["base"].str[-1]
            fasta_str = monomers["base"].str.cat()
            kmers = self._extract_kmers(fasta_str)

        if self.modification_abundance == True:
            if self.split_strands:
                for i, strand in enumerate(monomers["polymer"].unique()):
                    monomers_strand = monomers.loc[monomers["polymer"] == strand].copy()
                    sugar_mod_counts = self._count_modifications(
                        monomers_strand["sugar"]
                    )
                    sugar_mod_counts = {
                        f"RNA{i+1}_{k}": v for k, v in sugar_mod_counts.items()
                    }
                    phosphate_mod_counts = self._count_modifications(
                        monomers_strand["phosphate"]
                    )
                    phosphate_mod_counts = {
                        f"RNA{i+1}_{k}": v for k, v in phosphate_mod_counts.items()
                    }
                    kmers = merge_dicts(kmers, sugar_mod_counts, phosphate_mod_counts)
            else:
                sugar_mod_counts = self._count_modifications(monomers["sugar"])
                phosphate_mod_counts = self._count_modifications(monomers["phosphate"])
                kmers = merge_dicts(kmers, sugar_mod_counts, phosphate_mod_counts)
        return kmers

    def _extract_features_test(self, oligo_helm: str) -> dict:
        """
        Extracts features from an oligonucleotide HELM string.
        This method processes an oligonucleotide HELM string to extract k-mer features and optionally modification abundances.

        Args:
            oligo_helm (str): The HELM string representing the oligonucleotide.

        Returns:
            dict: A dictionary containing k-mer counts and optionally modification counts.

        Raises:
            AssertionError: If no k-mer entries are available from the training data.

        Notes:
            - The method assumes that the `kmer_entries` attribute is populated with k-mers from the training data.
            - A quick fix is applied to handle modified bases, which should be replaced with a proper HELM to FASTA conversion in the future.
            - If `modification_abundance` is set to True, the method also counts sugar and phosphate modifications and includes them in the returned dictionary.
        """
        assert len(self.kmer_entries) > 0, "No kmers from training data"
        kmer_entries_set = set(feature.split("_")[0] for feature in self.kmer_entries)
        monomers = self._extract_monomers(oligo_helm)
        # Added a quick fix to handle modified bases 
        if self.split_strands:
            fasta_str = []
            for strand in monomers["polymer"].unique():
                monomers_strand = monomers.loc[monomers["polymer"] == strand].copy()
                monomers_strand["base"] = monomers_strand["base"].str[-1]
                fasta_str.append(monomers_strand["base"].str.cat())
            kmers = defaultdict(int)
            for i, strand in enumerate(monomers["polymer"].unique()):
                for kmer in kmer_entries_set:
                    kmers[f"RNA{i+1}_{kmer}"] = count_overlapping(fasta_str[i], kmer)
        else:
            monomers["base"] = monomers["base"].str[-1]
            fasta_str = monomers["base"].str.cat()
            kmers = defaultdict(int)
            for i in kmer_entries_set:
                kmers[i] = count_overlapping(fasta_str, i)

        if self.modification_abundance == True:
            if self.split_strands:
                for i, strand in enumerate(monomers["polymer"].unique()):
                    monomers_strand = monomers.loc[monomers["polymer"] == strand].copy()
                    sugar_mod_counts = self._count_modifications(
                        monomers_strand["sugar"]
                    )
                    sugar_mod_counts = {
                        f"RNA{i+1}_{k}": v for k, v in sugar_mod_counts.items()
                    }
                    phosphate_mod_counts = self._count_modifications(
                        monomers_strand["phosphate"]
                    )
                    phosphate_mod_counts = {
                        f"RNA{i+1}_{k}": v for k, v in phosphate_mod_counts.items()
                    }
                    kmers.update(sugar_mod_counts)
                    kmers.update(phosphate_mod_counts)
            else:
                sugar_mod_counts = self._count_modifications(monomers["sugar"])
                phosphate_mod_counts = self._count_modifications(monomers["phosphate"])
                kmers.update(sugar_mod_counts)
                kmers.update(phosphate_mod_counts)
        return kmers

    def fit_transform(self, oligo_list: list) -> pd.DataFrame:
        """
        Extracts features from a list of oligonucleotides and returns them as a DataFrame.
        This method processes a list of oligonucleotides, extracts features for each oligonucleotide,
        and returns these features in a pandas DataFrame. Missing values are filled with zeroes,
        and all values are cast to integers.

        Args:
            oligo_list (list): A list of oligonucleotide sequences.

        Returns:
            pd.DataFrame: A DataFrame containing the extracted features for each oligonucleotide.
        """
        features = [self._extract_features(oligo) for oligo in oligo_list]
        features = pd.DataFrame(features).fillna(0).astype(int)
        self.kmer_entries = features.columns
        return features

    def transform(self, oligo_list: list) -> pd.DataFrame:
        """
        Transforms a list of oligonucleotides into a DataFrame of features. Subsets the features to only include k-mers from the training data.

        Args:
            oligo_list (list): A list of oligonucleotide sequences.

        Returns:
            pd.DataFrame: A DataFrame containing the extracted features for each oligonucleotide.
        """

        features = [self._extract_features_test(oligo) for oligo in oligo_list]
        features = pd.DataFrame(features).fillna(0).astype(int)
        features = features.reindex(columns=self.kmer_entries, fill_value=0)
        return features


class Thermodynamics:
    """
    A class used to perform thermodynamic calculations on RNA and hybrid RNA-DNA sequences.
    """

    def __init__(self):
        """
        Initializes the Thermodynamics class with nearest-neighbor parameters for RNA and hybrid RNA-DNA sequences.
        """

    def _get_fasta(self, helm: str) -> str:
        """
        Convert a HELM notation sequence to a FASTA sequence.

        Args:
            helm (str): The HELM notation sequence.

        Returns:
            str: The corresponding FASTA sequence.
        """
        xna = helm2xna(helm)
        xna = xna.polymers["RNA1"]["base"]
        fasta_str = "".join([i[-1] for i in xna.split(".")])
        return fasta_str

    def _prepare_sequence(self, fasta_str: str, nearest_neighbors_params: str) -> tuple:
        """
        Prepare the sequence for thermodynamic calculations.

        Args:
            fasta_str (str): The FASTA sequence.
            nearest_neighbors_params (str): The type of nearest-neighbor parameters to use.

        Returns:
            tuple: A tuple containing the prepared sequence and the corresponding nearest-neighbor parameters.
        """
        if nearest_neighbors_params == "RNA":
            nearest_neighbors_dict = DG_RNA
            fasta_str = fasta_str.replace("T", "U")
        elif nearest_neighbors_params == "hybrid":
            nearest_neighbors_dict = DG_HYBRID
            fasta_str = fasta_str.replace("U", "T")
        return fasta_str, nearest_neighbors_dict

    def _calculate_differential_stability(
        self, fasta_str: str, nearest_neighbors_params: str = "RNA"
    ) -> float:
        """
        Calculate the differential stability of an RNA sequence.

        Args:
            fasta_str (str): The FASTA sequence.
            nearest_neighbors_params (str, optional): The type of nearest-neighbor parameters to use. Defaults to "RNA".

        Returns:
            float: The differential stability of the sequence.
        """
        fasta_str, nearest_neighbors_dict = self._prepare_sequence(
            fasta_str, nearest_neighbors_params
        )
        start_pair = fasta_str[:2]
        end_pair = fasta_str[-2:]
        dG_start = nearest_neighbors_dict.get(start_pair, 0)
        dG_end = nearest_neighbors_dict.get(end_pair, 0)
        delta_dG = dG_end - dG_start
        return delta_dG

    def _calculate_dG(
        self,
        fasta_str: str,
        nearest_neighbors_params: str = "RNA",
        max_length: int = None,
    ) -> tuple:
        """
        Calculate the free energy change (dG) of an RNA sequence.

        Args:
            fasta_str (str): The FASTA sequence.
            nearest_neighbors_params (str, optional): The type of nearest-neighbor parameters to use. Defaults to "RNA".
            max_length (int, optional): The maximum length to calculate dG values for. Defaults to None.

        Returns:
            tuple: A tuple containing the total dG and a dictionary of basepair-specific dG values.
        """
        fasta_str, nearest_neighbors_dict = self._prepare_sequence(
            fasta_str, nearest_neighbors_params
        )
        dG_values = [
            nearest_neighbors_dict.get(fasta_str[i : i + 2], 0)
            for i in range(0, len(fasta_str) - 1)
        ]

        if max_length is not None:
            dG_values += [0] * (max_length - len(dG_values))

        basepair_dG = {f"dG_{i+1}_{i+2}": dG for i, dG in enumerate(dG_values)}

        intermolecular_initiation = 0
        if nearest_neighbors_params == "RNA":
            intermolecular_initiation = nearest_neighbors_dict["init"]
            au_end_penalty = (
                nearest_neighbors_dict["endAU"]
                if fasta_str[0] in ["A", "U"] or fasta_str[-1] in ["A", "U"]
                else 0
            )
            symmetry_correction = nearest_neighbors_dict["sym"]
            intermolecular_initiation += au_end_penalty + symmetry_correction
        elif nearest_neighbors_params == "hybrid":
            intermolecular_initiation = (
                nearest_neighbors_dict["init_CG"]
                if fasta_str[0] in ["C", "G"] or fasta_str[-1] in ["C", "G"]
                else 0
            )
            intermolecular_initiation += (
                nearest_neighbors_dict["init_TA"]
                if fasta_str[0] in ["A", "T"] and fasta_str[-1] in ["A", "T"]
                else 0
            )

        total_dG = intermolecular_initiation + sum(dG_values)

        return total_dG, basepair_dG

    def _calculate_mfe(self, fasta_str: str, target_mrna: str = None) -> float:
        """
        Calculate the minimum free energy (MFE) of an RNA sequence.

        Args:
            fasta_str (str): The FASTA sequence.
            target_mrna (str, optional): The target mRNA sequence. Defaults to None.

        Returns:
            float: The MFE of the sequence.
        """
        sequence = fasta_str + "&" + target_mrna if target_mrna else fasta_str
        fc = RNA.fold_compound(sequence)
        _, mfe = fc.mfe()
        return mfe

    def _extract_features(
        self,
        oligo_helm: str,
        target_mrna: str = None,
        nearest_neighbors_params: str = "RNA",
        max_length: int = None,
    ) -> dict:
        """
        Extract features from an RNA sequence.

        Args:
            oligo_helm (str): The HELM notation sequence.
            target_mrna (str, optional): The target mRNA sequence. Defaults to None.
            nearest_neighbors_params (str, optional): The type of nearest-neighbor parameters to use. Defaults to "RNA".
            max_length (int, optional): The maximum length to calculate dG values for. Defaults to None.

        Returns:
            dict: A dictionary containing the extracted features.
        """
        fasta_str = self._get_fasta(oligo_helm)
        basepair_dG = self._calculate_dG(
            fasta_str,
            nearest_neighbors_params=nearest_neighbors_params,
            max_length=max_length,
        )[1]
        return {
            "duplex_MFE": (
                self._calculate_mfe(fasta_str, target_mrna) if target_mrna else None
            ),
            "target_MFE": self._calculate_mfe(target_mrna) if target_mrna else None,
            "self_MFE": self._calculate_mfe(fasta_str),
            "dG_total": self._calculate_dG(
                fasta_str,
                nearest_neighbors_params=nearest_neighbors_params,
                max_length=max_length,
            )[0],
            "delta_dG": self._calculate_differential_stability(
                fasta_str, nearest_neighbors_params=nearest_neighbors_params
            ),
            **basepair_dG,
        }

    def fit_transform(
        self,
        oligo_list: list[str],
        target_mrnas: list[str] = None,
        nearest_neighbors_params: str = "RNA",
        pad_length: int = None,
        pad_value: float = 0,
    ) -> pd.DataFrame:
        """
        Featurize a dataset of RNA sequences.

        Args:
            oligo_list (list[str]): A list of HELM notation sequences.
            target_mrnas (list[str], optional): A list of target mRNA sequences. Defaults to None.
            nearest_neighbors_params (str, optional): The type of nearest-neighbor parameters to use. Defaults to "RNA".
            pad_length (int, optional): The length to pad the DataFrame to. Defaults to None.
            pad_value (float, optional): The value to use for padding. Defaults to 0.

        Returns:
            pandas.DataFrame: A DataFrame containing the extracted features.
        """
        max_length = max(len(self._get_fasta(seq)) for seq in oligo_list) - 1
        if pad_length is not None:
            max_length = max(max_length, pad_length - 1)

        if target_mrnas is None:
            features_list = [
                self._extract_features(
                    oligo_helm,
                    nearest_neighbors_params=nearest_neighbors_params,
                    max_length=max_length,
                )
                for oligo_helm in oligo_list
            ]
            features = pd.DataFrame(features_list)
            features = features.drop(columns=["duplex_MFE", "target_MFE"])
        else:
            features_list = [
                self._extract_features(
                    oligo_helm,
                    target_mrna,
                    nearest_neighbors_params=nearest_neighbors_params,
                    max_length=max_length,
                )
                for oligo_helm, target_mrna in zip(oligo_list, target_mrnas)
            ]
            features = pd.DataFrame(features_list)
        features.dropna(axis=1, inplace=True)

        for i in range(max_length):
            col_name = f"dG_{i+1}_{i+2}"
            if col_name not in features.columns:
                features[col_name] = pad_value

        return features

    def transform(
        self,
        oligo_list: list[str],
        target_mrnas: list[str] = None,
        nearest_neighbors_params: str = "RNA",
        pad_length: int = None,
        pad_value: float = 0,
    ) -> pd.DataFrame:
        """
        Featurize a dataset of RNA sequences.

        Args:
            oligo_list (list[str]): A list of HELM notation sequences.
            target_mrnas (list[str], optional): A list of target mRNA sequences. Defaults to None.
            nearest_neighbors_params (str, optional): The type of nearest-neighbor parameters to use. Defaults to "RNA".
            pad_length (int, optional): The length to pad the DataFrame to. Defaults to None.
            pad_value (float, optional): The value to use for padding. Defaults to 0.

        Returns:
            pandas.DataFrame: A DataFrame containing the extracted features.
        """
        return self.fit_transform(
            oligo_list, target_mrnas, nearest_neighbors_params, pad_length, pad_value
        )


class TargetDescriptors:
    """
    TargetDescriptors is a featurizer class that extracts mRNA target-specific descriptors
    from the given oligos.

    Args:
        modality (str): The modality of the target (e.g., 'ASO').
        email (str): Email address to be used with NCBI Entrez.
        db (str): Database to fetch sequences from, default is 'sequences'.
    """

    def __init__(self, modality: str, email: str = None, db: str = "sequences"):
        """
        Initializes the TargetDescriptors with the given parameters.
        """
        assert email is not None, "Email address is required for Entrez access"
        self.modality = modality
        self.email = email
        self.db = db

    def _get_entrez_seq(self, accession_id: str) -> str:
        """
        Fetches the sequence for a given NCBI accession id.

        Args:
            accession_id (str): NCBI accession id of the target sequence.

        Returns:
            str: The sequence corresponding to the given accession id.

        Raises:
            ValueError: If the accession id is not compatible with Entrez.
        """
        Entrez.email = self.email
        try:
            handle = Entrez.efetch(
                db=self.db, id=accession_id, rettype="fasta", retmode="text"
            )
            fasta_record = SeqIO.read(handle, "fasta")
            sequence = str(fasta_record.seq)
        except Exception as e:
            raise ValueError(
                f"Error fetching sequence for accession id {accession_id}: {e}"
            )

        return sequence

    def _get_sense_fasta(self, helm: str, strand: str) -> str:
        """
        Convert a HELM notation sequence of sense strand to a DNA FASTA sequence.

        Args:
            helm (str): The HELM notation sequence.
            strand (str): The strand identifier.

        Returns:
            str: The corresponding FASTA sequence.
        """
        xna = helm2xna(helm)
        xna = xna.polymers[strand]["base"]
        fasta_str = "".join([i[-1] for i in xna.split(".")])
        return fasta_str.upper().replace("U", "T")

    def _get_tiled_targets(self, target_site_length: int, target_seq: str) -> dict:
        """
        Tiles the entire target sequence given the target site length.

        Args:
            target_site_length (int): The length of each target site.
            target_seq (str): The mRNA target sequence.

        Returns:
            dict: A dictionary with the sequence of the tiled target as the key and the start position as the value.
        """
        tiled_targets = {}
        for i in range(len(target_seq) - target_site_length + 1):
            tiled_targets[target_seq[i : i + target_site_length]] = i
        return tiled_targets

    def _get_flanking_seqs(
        self,
        start_site: int,
        target_site_length: int,
        target_seq: str,
        flanking_length: int = 50,
    ) -> tuple:
        """
        Retrieves the flanking sequences given the target sequence, length of the target site, and the start site.

        Args:
            start_site (int): The start position of the target site within the target sequence.
            target_site_length (int): The length of the target site.
            target_seq (str): The mRNA target sequence.
            flanking_length (int, optional): The length of the flanking sequences to retrieve, default is 50.

        Returns:
            tuple: A tuple containing the upstream and downstream flanking sequences.
        """
        upstream_start = max(0, start_site - flanking_length)
        upstream_end = start_site
        downstream_start = start_site + target_site_length
        downstream_end = min(len(target_seq), downstream_start + flanking_length)
        upstream_flank = target_seq[upstream_start:upstream_end]
        downstream_flank = target_seq[downstream_start:downstream_end]
        return upstream_flank, downstream_flank

    def _get_local_mrna_structure(self, local_mrna_seq: str) -> tuple:
        """
        Predicts the secondary structure and minimum free energy (MFE) of a local mRNA sequence.

        Args:
            local_mRNA_seq (str): The local mRNA sequence for which the secondary structure and MFE are to be predicted.

        Returns:
            tuple: A tuple containing the predicted secondary structure (dot-bracket notation) and the MFE.
        """
        fold = RNA.fold(local_mrna_seq)
        return fold

    def _get_kmer_frequencies(self, target_sequence: str) -> pd.DataFrame:
        """
        Calculates the 1-mer, 2-mer, and 3-mer frequencies of a DNA target sequence and combines them into a DataFrame.

        Args:
            target_sequence (str): The DNA sequence.

        Returns:
            pd.DataFrame: A DataFrame containing the frequencies of 1-mers, 2-mers, and 3-mers.
        """

        def kmer_count(sequence: str, k: int) -> dict:
            bases = ["A", "T", "C", "G"]
            kmers = ["".join(p) for p in product(bases, repeat=k)]
            freq_dict = {kmer: 0 for kmer in kmers}

            kmer_counts = Counter(
                [sequence[i : i + k] for i in range(len(sequence) - k + 1)]
            )

            freq_dict.update(kmer_counts)
            return freq_dict

        one_mer_freq = kmer_count(target_sequence, 1)
        two_mer_freq = kmer_count(target_sequence, 2)
        three_mer_freq = kmer_count(target_sequence, 3)
        combined_freq = {**one_mer_freq, **two_mer_freq, **three_mer_freq}
        kmer_df = pd.DataFrame.from_dict(combined_freq, orient="index").transpose()
        return kmer_df

    def transform(
        self,
        oligo_list: list,
        targets: list,
        strand: str = "RNA1",
        flanking_length: int = 50,
        numerical: bool = False,
    ) -> pd.DataFrame:
        """
        Transforms the input data into target-specific descriptors.

        Args:
            X (list): The input data.
            targets (list): List of mRNA targets NCBI accession ids.
            strand (str): The strand identifier, default is "RNA1".
            flanking_length (int): The length of the flanking sequences to retrieve, default is 50.
            numerical (bool): Whether to return numerical features, default is False.

        Returns:
            pd.DataFrame: The transformed data.
        """
        oligo_fasta_seqs = [
            self._get_sense_fasta(oligo_helm, strand) for oligo_helm in oligo_list
        ]
        if self.modality.lower() == "aso":
            oligo_fasta_seqs = [
                str(Seq(seq).reverse_complement()) for seq in oligo_fasta_seqs
            ]

        oligo_lengths = list(map(len, oligo_fasta_seqs))
        upstream_flanks = []
        downstream_flanks = []
        local_target_structs = []
        local_target_mfes = []

        for oligo, length, target in zip(oligo_fasta_seqs, oligo_lengths, targets):
            target_seq = self.targets_dict[target]
            tiled_targets = self._get_tiled_targets(length, target_seq)
            closest_match, _ = process.extractOne(oligo, tiled_targets.keys())
            start_target_site = tiled_targets[closest_match]
            upstream_flank, downstream_flank = self._get_flanking_seqs(
                start_target_site, length, target_seq, flanking_length=flanking_length
            )
            upstream_flanks.append(upstream_flank)
            downstream_flanks.append(downstream_flank)
            local_target_struct, local_target_mfe = self._get_local_mrna_structure(
                upstream_flank + oligo + downstream_flank
            )
            local_target_structs.append(local_target_struct)
            local_target_mfes.append(local_target_mfe)

        if numerical:
            numerical_target_features = []
            for upstream_flank, downstream_flank in zip(
                upstream_flanks, downstream_flanks
            ):
                numerical_target_features.append(
                    pd.concat(
                        [
                            self._get_kmer_frequencies(upstream_flank),
                            self._get_kmer_frequencies(downstream_flank),
                        ],
                        axis=1,
                    )
                )
            numerical_target_features = pd.concat(
                numerical_target_features, axis=0, ignore_index=True
            )
            numerical_target_features["local_target_MFE"] = local_target_mfes
            return numerical_target_features
        else:
            target_features = pd.DataFrame(
                {
                    "target_id": targets,
                    "upstream_flank": upstream_flanks,
                    "downstream_flank": downstream_flanks,
                    "local_target_MFE": local_target_mfes,
                    "local_target_structure": local_target_structs,
                }
            )
            return target_features

    def fit_transform(
        self,
        oligo_list: list,
        targets: list,
        strand: str = "RNA1",
        flanking_length: int = 50,
        numerical: bool = False,
    ) -> pd.DataFrame:
        """
        Fits the featurizer to the data and then transforms it.

        Args:
            X (list): The input data.
            targets (list): List of mRNA targets NCBI accession ids.
            strand (str): The strand identifier, default is "RNA1".
            flanking_length (int): The length of the flanking sequences to retrieve, default is 50.
            numerical (bool): Whether to return numerical features, default is False.

        Returns:
            pd.DataFrame: The transformed data.
        """
        target_acc_lists = list(set(targets))
        targets_sequences = [self._get_entrez_seq(acc) for acc in target_acc_lists]
        self.targets_dict = dict(zip(target_acc_lists, targets_sequences))

        return self.transform(
            oligo_list,
            targets,
            strand=strand,
            flanking_length=flanking_length,
            numerical=numerical,
        )
