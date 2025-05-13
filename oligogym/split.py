import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.preprocessing import normalize

from .helm import helm2xna
from .features import KMersCounts


def random_split(
    x,
    y,
    test_size,
    val_size=None,
    random_state=None,
    return_index=False,
    stratified=False,
):
    """
    Perform a random train/test/validation split on the dataset.

    Args:
        x (array-like): Features.
        y (array-like): Labels.
        test_size (float): The proportion of the dataset to include in the test split.
        val_size (float, optional): The proportion of the dataset to include in the validation split. Defaults to None.
        random_state (int, optional): The seed used by the random number generator. Defaults to None.
        return_index (bool, optional): Whether to return the indices of the train/test split. Defaults to False.
        stratified (bool, optional): Whether to perform stratified sampling. Defaults to False.

    Returns:
        tuple: The train/validation/test split of the dataset, and optionally the indices of the split.
            - If `val_size` is provided and `return_index` is True:
                (X_train, X_val, X_test, y_train, y_val, y_test, train_idx, val_idx, test_idx)
            - If `val_size` is provided and `return_index` is False:
                (X_train, X_val, X_test, y_train, y_val, y_test)
            - If `val_size` is not provided and `return_index` is True:
                (X_train, X_test, y_train, y_test, train_idx, test_idx)
            - If `val_size` is not provided and `return_index` is False:
                (X_train, X_test, y_train, y_test)
    """
    stratify = y if stratified else None
    X_train, X_temp, y_train, y_temp, train_idx, temp_idx = train_test_split(
        x,
        y,
        range(len(x)),
        test_size=test_size + (val_size or 0),
        random_state=random_state,
        stratify=stratify,
    )

    if val_size:
        val_size_adjusted = val_size / (test_size + val_size)
        X_val, X_test, y_val, y_test, val_idx, test_idx = train_test_split(
            X_temp,
            y_temp,
            temp_idx,
            test_size=val_size_adjusted,
            random_state=random_state,
            stratify=stratify,
        )
        if return_index:
            return (
                X_train,
                X_val,
                X_test,
                y_train,
                y_val,
                y_test,
                train_idx,
                val_idx,
                test_idx,
            )
        else:
            return X_train, X_val, X_test, y_train, y_val, y_test
    else:
        X_test, y_test, test_idx = X_temp, y_temp, temp_idx
        if return_index:
            return X_train, X_test, y_train, y_test, train_idx, test_idx
        else:
            return X_train, X_test, y_train, y_test


def target_split(
    x, y, targets, test_size, val_size=None, random_state=None, return_index=False
):
    """
    Perform a train/test/validation split on the dataset based on the mRNA target labels.

    Args:
        x (array-like): Features.
        y (array-like): Labels.
        targets (array-like): Target labels for grouping.
        test_size (float): The proportion of the dataset to include in the test split.
        val_size (float, optional): The proportion of the dataset to include in the validation split. Defaults to None.
        random_state (int, optional): The seed used by the random number generator. Defaults to None.
        return_index (bool, optional): Whether to return the indices of the train/test split. Defaults to False.

    Returns:
        tuple: The train/validation/test split of the dataset, and optionally the indices of the split.
            - If `val_size` is provided and `return_index` is True:
                (X_train, X_val, X_test, y_train, y_val, y_test, train_idx, val_idx, test_idx)
            - If `val_size` is provided and `return_index` is False:
                (X_train, X_val, X_test, y_train, y_val, y_test)
            - If `val_size` is not provided and `return_index` is True:
                (X_train, X_test, y_train, y_test, train_idx, test_idx)
            - If `val_size` is not provided and `return_index` is False:
                (X_train, X_test, y_train, y_test)
    """
    nan_mask = targets.isna()
    groups = np.where(nan_mask, "unk", targets)

    gss = GroupShuffleSplit(
        n_splits=1, test_size=test_size + (val_size or 0), random_state=random_state
    )
    train_idx, temp_idx = next(gss.split(x, y, groups))
    X_train, X_temp = x[train_idx], x[temp_idx]
    y_train, y_temp = y[train_idx], y[temp_idx]

    if val_size:
        val_size_adjusted = val_size / (test_size + val_size)
        gss_val = GroupShuffleSplit(
            n_splits=1, test_size=val_size_adjusted, random_state=random_state
        )
        val_idx, test_idx = next(gss_val.split(X_temp, y_temp, groups[temp_idx]))
        X_val, X_test = X_temp[val_idx], X_temp[test_idx]
        y_val, y_test = y_temp[val_idx], y_temp[test_idx]
        if return_index:
            return (
                X_train,
                X_val,
                X_test,
                y_train,
                y_val,
                y_test,
                train_idx,
                val_idx,
                test_idx,
            )
        else:
            return X_train, X_val, X_test, y_train, y_val, y_test
    else:
        X_test, y_test, test_idx = X_temp, y_temp, temp_idx
        if return_index:
            return X_train, X_test, y_train, y_test, train_idx, test_idx
        else:
            return X_train, X_test, y_train, y_test


def backbone_split(
    x,
    y,
    test_size,
    val_size=None,
    random_state=None,
    return_index=False,
    strand="RNA1",
):
    """
    Perform a train/test/validation split on the dataset based on the backbone design of the HELM sequences.

    Args:
        x (array-like): Features.
        y (array-like): Labels.
        test_size (float): The proportion of the dataset to include in the test split.
        val_size (float, optional): The proportion of the dataset to include in the validation split. Defaults to None.
        random_state (int, optional): The seed used by the random number generator. Defaults to None.
        return_index (bool, optional): Whether to return the indices of the train/test split. Defaults to False.
        strand (str, optional): Specify which RNA strand to use for backbone patterns detection. "both" will combine patterns from RNA1 and RNA2 strands.

    Returns:
        tuple: The train/validation/test split of the dataset, and optionally the indices of the split.
            - If `val_size` is provided and `return_index` is True:
                (X_train, X_val, X_test, y_train, y_val, y_test, train_idx, val_idx, test_idx)
            - If `val_size` is provided and `return_index` is False:
                (X_train, X_val, X_test, y_train, y_val, y_test)
            - If `val_size` is not provided and `return_index` is True:
                (X_train, X_test, y_train, y_test, train_idx, test_idx)
            - If `val_size` is not provided and `return_index` is False:
                (X_train, X_test, y_train, y_test)
    """

    def get_backbone_design(oligo_helm, strand="RNA1"):
        """
        Extract the backbone design from a given HELM sequence.

        Args:
            oligo_helm (str): The HELM sequence.

        Returns:
            str: The backbone design of the HELM sequence.
        """
        xna = helm2xna(oligo_helm)
        sugar = xna.polymers[strand]["sugar"]
        phosphate = xna.polymers[strand]["phosphate"]
        return sugar + phosphate

    if strand == "both":
        backbone_designs_1 = [get_backbone_design(helm_seq, "RNA1") for helm_seq in x]

        backbone_designs_2 = [get_backbone_design(helm_seq, "RNA2") for helm_seq in x]

        backbone_designs = [
            i + j for i, j in zip(backbone_designs_1, backbone_designs_2)
        ]

    else:
        backbone_designs = [get_backbone_design(helm_seq, strand) for helm_seq in x]

    backbone_designs = np.array(backbone_designs)

    gss = GroupShuffleSplit(
        n_splits=1, test_size=test_size + (val_size or 0), random_state=random_state
    )
    train_idx, temp_idx = next(gss.split(x, y, backbone_designs))
    X_train, X_temp = x[train_idx], x[temp_idx]
    y_train, y_temp = y[train_idx], y[temp_idx]

    if val_size:
        val_size_adjusted = val_size / (test_size + val_size)
        gss_val = GroupShuffleSplit(
            n_splits=1, test_size=val_size_adjusted, random_state=random_state
        )
        val_idx, test_idx = next(
            gss_val.split(X_temp, y_temp, backbone_designs[temp_idx])
        )
        X_val, X_test = X_temp[val_idx], X_temp[test_idx]
        y_val, y_test = y_temp[val_idx], y_temp[test_idx]
        if return_index:
            return (
                X_train,
                X_val,
                X_test,
                y_train,
                y_val,
                y_test,
                train_idx,
                val_idx,
                test_idx,
            )
        else:
            return X_train, X_val, X_test, y_train, y_val, y_test
    else:
        X_test, y_test, test_idx = X_temp, y_temp, temp_idx
        if return_index:
            return X_train, X_test, y_train, y_test, train_idx, test_idx
        else:
            return X_train, X_test, y_train, y_test


def nucleobase_split(
    x,
    y,
    test_size,
    val_size=None,
    random_state=None,
    return_index=False,
    kmer_max=3,
    kmer=True,
    strand="RNA1",
    **kwargs,
):
    """
    Perform a train/test/validation split on the dataset based on the nucleobase composition of the HELM sequences.

    This method first transforms the HELM sequences into k-mer vectors (with k=1,..,kmer_max), then applies KMeans clustering
    to these vectors. The resulting cluster labels are used as groups for the GroupShuffleSplit.

    Args:
        x (array-like): Features.
        y (array-like): Labels.
        test_size (float): The proportion of the dataset to include in the test split.
        val_size (float, optional): The proportion of the dataset to include in the validation split. Defaults to None.
        random_state (int, optional): The seed used by the random number generator. Defaults to None.
        return_index (bool, optional): Whether to return the indices of the train/test split. Defaults to False.
        kmer_max (int, optional): The maximum length of the kmer to use for creating the kmer frequency matrix. Defaults to 3.
        kmer (bool, optional): Whether to use kmer frequency or fasta identity as label for splitting. Defaults to True.
        strand (str, optional): Specify which RNA strand to use for nucleobase patterns detection. "both" will combine patterns from RNA1 and RNA2 strands. Only relevant when kmer == False.
        **kwargs: Additional keyword arguments to pass to the KMeans constructor.

    Returns:
        tuple: The train/validation/test split of the dataset, the indices of the split (if return_index is True),
            and the labels of the KMeans clustering.
            - If `val_size` is provided and `return_index` is True:
                (X_train, X_val, X_test, y_train, y_val, y_test, train_idx, val_idx, test_idx, group_labels)
            - If `val_size` is provided and `return_index` is False:
                (X_train, X_val, X_test, y_train, y_val, y_test, group_labels)
            - If `val_size` is not provided and `return_index` is True:
                (X_train, X_test, y_train, y_test, train_idx, test_idx, group_labels)
            - If `val_size` is not provided and `return_index` is False:
                (X_train, X_test, y_train, y_test, group_labels)
    """

    def get_fasta(helm: str, strand="RNA1") -> str:
        """
        Convert a HELM notation sequence to a FASTA sequence.

        Args:
            helm (str): The HELM notation sequence.

        Returns:
            str: The corresponding FASTA sequence.
        """
        xna = helm2xna(helm)
        xna = xna.polymers[strand]["base"]
        fasta_str = "".join([i[-1] for i in xna.split(".")])
        return fasta_str

    if kmer:
        kmers = KMersCounts(k=list(range(1, kmer_max + 1)))
        kmers_vectors = np.array(kmers.fit_transform(x))
        kmers_vectors = normalize(kmers_vectors, norm="l2")
        kmeans = KMeans(random_state=random_state, **kwargs).fit(kmers_vectors)
        group_labels = kmeans.labels_
    else:
        if strand == "both":
            fasta_1 = [get_fasta(helm_seq, "RNA1") for helm_seq in x]

            fasta_2 = [get_fasta(helm_seq, "RNA2") for helm_seq in x]

            group_labels = [i + j for i, j in zip(fasta_1, fasta_2)]

        else:
            group_labels = [get_fasta(helm_seq, strand) for helm_seq in x]

    gss = GroupShuffleSplit(
        n_splits=1, test_size=test_size + (val_size or 0), random_state=random_state
    )
    train_idx, temp_idx = next(gss.split(x, y, group_labels))
    X_train, X_temp = x[train_idx], x[temp_idx]
    y_train, y_temp = y[train_idx], y[temp_idx]

    if val_size:
        val_size_adjusted = val_size / (test_size + val_size)
        gss_val = GroupShuffleSplit(
            n_splits=1, test_size=val_size_adjusted, random_state=random_state
        )
        val_idx, test_idx = next(
            gss_val.split(X_temp, y_temp, kmeans.labels_[temp_idx])
        )
        X_val, X_test = X_temp[val_idx], X_temp[test_idx]
        y_val, y_test = y_temp[val_idx], y_temp[test_idx]
        if return_index:
            return (
                X_train,
                X_val,
                X_test,
                y_train,
                y_val,
                y_test,
                train_idx,
                val_idx,
                test_idx,
            )
        else:
            return X_train, X_val, X_test, y_train, y_val, y_test
    else:
        X_test, y_test, test_idx = X_temp, y_temp, temp_idx
        if return_index:
            return X_train, X_test, y_train, y_test, train_idx, test_idx
        else:
            return X_train, X_test, y_train, y_test


def time_split(
    x,
    y,
    timestamp,
    test_size=None,
    val_size=None,
    return_index=False,
    cutoff_date=None,
):
    """
    Perform a train/test/validation split on the dataset based on time.

    Args:
        x (array-like): Features.
        y (array-like): Labels.
        timestamp (array-like): Timestamp labels for sorting.
        test_size (float, optional): The proportion of the dataset to include in the test split. Defaults to None.
        val_size (float, optional): The proportion of the dataset to include in the validation split. Defaults to None.
        return_index (bool, optional): Whether to return the indices of the train/test split. Defaults to False.
        cutoff_date (np.datetime64 or list of np.datetime64, optional): The cutoff date(s) for splitting the data. Defaults to None.

    Returns:
        tuple: The train/validation/test split of the dataset, and optionally the indices of the split.
            - If `cutoff_date` is provided and `return_index` is True:
                (X_train, X_val, X_test, y_train, y_val, y_test, train_idx, val_idx, test_idx)
            - If `cutoff_date` is provided and `return_index` is False:
                (X_train, X_val, X_test, y_train, y_val, y_test)
            - If `cutoff_date` is not provided and `return_index` is True:
                (X_train, X_test, y_train, y_test, train_idx, test_idx)
            - If `cutoff_date` is not provided and `return_index` is False:
                (X_train, X_test, y_train, y_test)
    """
    try:
        timestamp = np.array(timestamp, dtype="datetime64")
    except ValueError as e:
        raise ValueError(
            "Invalid timestamp format. Ensure all timestamps are valid np.datetime64 objects."
        ) from e

    sorted_indices = np.argsort(timestamp)
    x, y, timestamp = x[sorted_indices], y[sorted_indices], timestamp[sorted_indices]

    if cutoff_date is not None:
        if isinstance(cutoff_date, (str, np.datetime64)):
            cutoff_date = [cutoff_date]
        cutoff_date = np.array(cutoff_date, dtype="datetime64")
        if len(cutoff_date) == 1:
            train_idx = timestamp < cutoff_date[0]
            test_idx = timestamp >= cutoff_date[0]
            X_train, X_test = x[train_idx], x[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            if return_index:
                return (
                    X_train,
                    X_test,
                    y_train,
                    y_test,
                    sorted_indices[train_idx],
                    sorted_indices[test_idx],
                )
            else:
                return X_train, X_test, y_train, y_test
        elif len(cutoff_date) == 2:
            train_idx = timestamp < cutoff_date[0]
            val_idx = (timestamp >= cutoff_date[0]) & (timestamp < cutoff_date[1])
            test_idx = timestamp >= cutoff_date[1]
            X_train, X_val, X_test = x[train_idx], x[val_idx], x[test_idx]
            y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]
            if return_index:
                return (
                    X_train,
                    X_val,
                    X_test,
                    y_train,
                    y_val,
                    y_test,
                    sorted_indices[train_idx],
                    sorted_indices[val_idx],
                    sorted_indices[test_idx],
                )
            else:
                return X_train, X_val, X_test, y_train, y_val, y_test
        else:
            raise ValueError(
                "cutoff_date must be a single timestamp or a list of two timestamps."
            )
    else:
        if test_size is None:
            raise ValueError("test_size must be provided if cutoff_date is not given.")

        n = len(x)
        test_end = int(n * (1 - test_size))
        if val_size:
            val_end = int(test_end * (1 - val_size))
        else:
            val_end = test_end

        X_train, X_val_test = x[:val_end], x[val_end:]
        y_train, y_val_test = y[:val_end], y[val_end:]

        if val_size:
            X_val, X_test = (
                X_val_test[: int(len(X_val_test) * val_size)],
                X_val_test[int(len(X_val_test) * val_size) :],
            )
            y_val, y_test = (
                y_val_test[: int(len(y_val_test) * val_size)],
                y_val_test[int(len(y_val_test) * val_size) :],
            )
            if return_index:
                return (
                    X_train,
                    X_val,
                    X_test,
                    y_train,
                    y_val,
                    y_test,
                    sorted_indices[:val_end],
                    sorted_indices[val_end : val_end + int(len(X_val_test) * val_size)],
                    sorted_indices[val_end + int(len(X_val_test) * val_size) :],
                )
            else:
                return X_train, X_val, X_test, y_train, y_val, y_test
        else:
            X_test, y_test = X_val_test, y_val_test
            if return_index:
                return (
                    X_train,
                    X_test,
                    y_train,
                    y_test,
                    sorted_indices[:val_end],
                    sorted_indices[val_end:],
                )
            else:
                return X_train, X_test, y_train, y_test
