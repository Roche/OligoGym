import random

import numpy as np
from .helm import helm2xna
from .features import KMersCounts
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize


def count_strands(helm):
    """
    Counts the number of RNA strands in the given HELM notation.

    Args:
        helm (str): The HELM notation string representing the oligonucleotide sequence.

    Returns:
        int: The number of RNA strands.
    """
    xna = helm2xna(helm)
    strands = [i for i in xna.polymers.keys() if "RNA" in i]
    num_strands = len(strands)
    return num_strands


def get_strands(helm):
    """
    Retrieves the RNA strands from the given HELM notation.

    Args:
        helm (str): The HELM notation string representing the oligonucleotide sequence.

    Returns:
        list: A list of RNA strand identifiers.
    """
    xna = helm2xna(helm)
    strands = [i for i in xna.polymers.keys() if "RNA" in i]
    return strands


def get_nt_seq_len(helm, strand="RNA1"):
    """
    Returns the length of an oligonucleotide sequence.

    Args:
        helm (str): The oligonucleotide sequence.

    Returns:
        int: The length of the oligonucleotide sequence.
    """
    xna = helm2xna(helm)
    xna = xna.polymers[strand]["base"]
    xna = list(filter(None, xna.split(".")))
    xna = [s for s in xna if s]
    xna = "".join([i[-1] for i in xna])
    nt_seq_len = len(xna)
    return nt_seq_len


def get_unique_monomers(helm, strand="RNA1"):
    """
    Returns a list of unique monomers for a given helm.

    Args:
        helm (str): The input helm.

    Returns:
        list: A list of unique monomers.
    """
    xna = helm2xna(helm)
    xna = xna.polymers[strand]
    base_monomers = list(filter(None, list(set(xna["base"].split(".")))))
    ribose_monomers = list(filter(None, list(set(xna["sugar"].split(".")))))
    phosphate_monomers = list(filter(None, list(set(xna["phosphate"].split(".")))))
    monomers = base_monomers + ribose_monomers + phosphate_monomers
    return list(filter(None, monomers))


def get_gc_content(helm, strand="RNA1"):
    """
    Calculate the GC content of a given oligonucleotide sequence.

    Args:
        helm (str): The oligonucleotide sequence.

    Returns:
        float: The GC content of the oligonucleotide sequence.
    """
    xna = helm2xna(helm)
    xna = xna.polymers[strand]["base"]
    xna = list(filter(None, xna.split(".")))
    xna = [s for s in xna if s]
    nucleotides = [i[-1] for i in xna]
    gc_count = nucleotides.count("G") + nucleotides.count("C")
    gc_content = gc_count / len(nucleotides) * 100

    return gc_content


def get_nt_content(helm, strand="RNA1"):
    """
    Calculate the nucleotide content of a given oligonucleotide sequence.

    Args:
        helm (str): The oligonucleotide sequence.

    Returns:
        tuple: The nucleotide content of the oligonucleotide sequence.
    """
    xna = helm2xna(helm)
    xna = xna.polymers[strand]["base"]
    xna = list(filter(None, xna.split(".")))
    xna = [s for s in xna if s]
    nucleotides = [i[-1] for i in xna]
    nucleotides = list(filter(None, nucleotides))
    g_content = nucleotides.count("G") / len(nucleotides) * 100
    c_content = nucleotides.count("C") / len(nucleotides) * 100
    a_content = nucleotides.count("A") / len(nucleotides) * 100
    tu_content = (
        (nucleotides.count("T") + nucleotides.count("U")) / len(nucleotides) * 100
    )

    return (g_content, c_content, a_content, tu_content)


def get_cosine_dist_to_nearest_neighbor(
    helm_list,
    helm_list2=None,
    kmer_max=5,
    strands=None,
    modification_abundance=False,
):
    """
    Calculate the cosine distance to the nearest neighbor.

    Args:
        helm_list (list): A list of helm sequences.
        helm_list2 (list, optional): A second list of helm sequences.
        kmer_max (int, optional): The maximum length of the kmer to use for creating the kmer frequency matrix. Defaults to 5.

    Returns:
        list: A list containing the cosine distance to the nearest neighbor.
    """
    kmers = KMersCounts(
        k=list(range(1, kmer_max + 1)),
        strands=strands,
        modification_abundance=modification_abundance,
    )

    if helm_list2 is not None:
        if isinstance(helm_list, np.ndarray):
            helm_list = helm_list.tolist()
        if isinstance(helm_list2, np.ndarray):
            helm_list2 = helm_list2.tolist()

        combined_list = helm_list + helm_list2
        combined_vectors = np.array(kmers.fit_transform(combined_list))
        combined_vectors = normalize(combined_vectors, norm="l2")

        kmers_vectors = combined_vectors[: len(helm_list)]
        kmers_vectors2 = combined_vectors[len(helm_list) :]
    else:
        kmers_vectors = np.array(kmers.fit_transform(helm_list))
        kmers_vectors = normalize(kmers_vectors, norm="l2")

    nn = NearestNeighbors(n_neighbors=2, metric="cosine")
    nn.fit(kmers_vectors)

    if helm_list2 is not None:
        distances, _ = nn.kneighbors(kmers_vectors2)
    else:
        distances, _ = nn.kneighbors(kmers_vectors)

    nearest_neighbor_distances = distances[:, 1]
    return nearest_neighbor_distances


def get_uniqueness(helm_list):
    """
    Returns a list of labels identifying unique helm sequences.

    Args:
        helm_list (list): A list of elements.

    Returns:
        list: A list containing the label for each helm sequence identified as unique within the list.
    """
    _, unique_id = np.unique(helm_list, return_index=True)
    uniqueness_labels = np.zeros(len(helm_list))
    uniqueness_labels.flat[unique_id] = 1

    return uniqueness_labels


def get_outliers(data, return_index=False):
    """
    Finds the outliers in a given dataset.

    Args:
        data (array-like): The input dataset.
        return_index (bool, optional): Whether to return the indices of the outliers. Default is False.

    Returns:
        array-like: The outliers in the dataset.
        array-like, optional: The indices of the outliers, only returned if `return_index` is True.
    """
    Q1 = np.percentile(data, 25, method="midpoint")
    Q3 = np.percentile(data, 75, method="midpoint")
    IQR = Q3 - Q1

    upper_limit = Q3 + 1.5 * IQR
    lower_limit = Q1 - 1.5 * IQR

    outlier_indices = np.where((data > upper_limit) | (data < lower_limit))

    if return_index:
        outliers = data[outlier_indices]
        return outliers, outlier_indices
    else:
        outliers = data[outlier_indices]
        return outliers


def get_xna(helm, strand="RNA1"):
    """
    Converts a HELM notation string to XNA format.

    Args:
        helm (str): A HELM notations string.

    Returns:
        tuple: A tuple containing the base, sugar, and phosphate components of RNA1 in XNA format.
    """
    xna = helm2xna(helm)
    xna = xna.polymers[strand]

    return (xna["base"], xna["sugar"], xna["phosphate"])
