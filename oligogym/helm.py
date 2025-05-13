from pprint import pformat
from typing import Dict, List

import pandas as pd

class XNAData:
    """
    Represents the data structure for XNA (human readable oligonucleotide representation) information.

    Args:
        polymers (Dict): A dictionary containing the polymers.
        connections (List): A list containing the connections between polymers.
        annotations (str): Additional annotations for the XNA data.
    """

    def __init__(
        self,
        polymers: Dict[str, Dict[str, str]] = None,
        connections: List[Dict[str, str]] = None,
        annotations: str = None,
    ):
        for polymer in polymers.keys():
            assert polymer.startswith(
                ("CHEM", "RNA")
            ), "Only CHEM and RNA polymers are allowed"
            assert all(
                polymer in ["sugar", "base", "phosphate", "chemistry"]
                for polymer in polymers[polymer].keys()
            ), "Polymers should contain 'sugar', 'base', 'phosphate' or 'chemistry' keys"
            assert (
                len(set([comp.count(".") for _, comp in polymers[polymer].items()]))
                == 1
            ), "Lengths of passed components should match"

        self.polymers = polymers
        self.connections = connections
        self.annotations = annotations

    def __str__(self):
        """
        Returns a formatted string representation of the XNAData object.
        """
        return pformat([self.polymers, self.connections, self.annotations])

    def __eq__(self, other):
        """
        Checks if the XNAData object is equal to another XNAData object.
        Args:
            other (XNAData): The other XNAData object to compare with.
        Returns:
            bool: True if the XNAData objects are equal, False otherwise.
        """
        return all(
            [
                self.polymers == other.polymers,
                self.connections == other.connections,
                self.annotations == other.annotations,
            ]
        )

    def to_helm(self):
        """
        Converts the XNAData object to the HELM format.

        This method converts the XNAData object to the HELM (Hierarchical Editing Language for Macromolecules) format.

        Returns:
            str: The HELM format representation of the XNAData object.

        """
        return xna2helm(self)

    @staticmethod
    def from_helm(helm):
        """
        Converts the HELM format to an XNAData object.

        This static method converts the HELM format to an XNAData object.

        Args:
            helm (str): The HELM format representation of the XNAData object.

        Returns:
            xna (XNAData): The XNAData object created from the HELM format.

        """
        return helm2xna(helm)
    

def helm2xna(helm):
    """
    Convert a HELM notation to an XNA notation.

    Args:
        helm (str): The input HELM notation.

    Returns:
        XNAData: A data structure representing the XNA notation.
    """
    (simple_polymers, connections, polymer_groups, annotations, helm_version) = (
        split_complex_polymer(helm)
    )

    # Process and parse polymers
    simple_polymers = split_simple_polymers(simple_polymers)
    polymer_dict = {}

    for polymer in simple_polymers:
        polymer_name = polymer.split("{")[0]
        assert polymer.startswith(
            ("CHEM", "RNA")
        ), "Only RNA and CHEM polymers are allowed"

        if polymer_name.startswith("RNA"):
            monomers = split_monomers(polymer)
            sugar_list = []
            base_list = []
            phosphate_list = []

            for monomer in monomers:
                sugar, base, phosphate = parse_monomer(monomer)
                sugar_list.append(sugar)
                base_list.append(base)
                phosphate_list.append(phosphate)

            sugar_string = ".".join(sugar_list)
            base_string = ".".join(base_list)
            phosphate_string = ".".join(phosphate_list)
            polymer_dict[polymer_name] = {
                "sugar": sugar_string,
                "base": base_string,
                "phosphate": phosphate_string,
            }

        elif polymer_name.startswith("CHEM"):
            chem_monomer = polymer.split("{")[1].split("}")[0]
            if "[" in chem_monomer:
                chem_monomer = chem_monomer.replace("[", "").replace("]", "")
            polymer_dict[polymer_name] = {"chemistry": chem_monomer}

    # Process and parse connections
    connection_list = []
    if connections:
        split_connections = connections.split("|")
        for connection in split_connections:
            polymer_1 = connection.split(",")[0]
            polymer_2 = connection.split(",")[1]
            connection_dict = {
                "polymer_1": polymer_1,
                "polymer_2": polymer_2,
                "connection": connection.split(",")[2],
            }
            connection_list.append(connection_dict)

    xna = XNAData(
        polymers=polymer_dict, connections=connection_list, annotations=annotations
    )
    return xna


def split_complex_polymer(helm):
    """
    Splits a complex polymer HELM string into its individual components.

    Args:
        helm (str): The complex polymer HELM string to be split.

    Returns:
        simple_polymers (str): The simple polymers part of the HELM string.
        connections (str): The connections part of the HELM string.
        polymer_groups (str): The polymer groups part of the HELM string.
        annotations (str): The annotations part of the HELM string.
        helm_version (str): The version of HELM used.

    Raises:
        AssertionError: If the HELM string does not contain exactly 4 '$' characters.
    """
    #assert helm.count("$") == 4, "HELM should contain 4 '$' characters"
    (simple_polymers, connections, polymer_groups, annotations, helm_version) = (
        helm.split("$")
    )
    return simple_polymers, connections, polymer_groups, annotations, helm_version


def split_simple_polymers(polymers):
    """
    Split a string of polymers separated by '|' into a list of individual polymers.

    Args:
        polymers (str): A string of polymers separated by '|'.

    Returns:
        split_polymers (list): A list of individual polymers.
    """
    split_polymers = polymers.split("|")
    return split_polymers


def split_monomers(polymer):
    """
    Splits the given polymer into individual monomers.

    Args:
        polymer (str): The polymer to split.

    Returns:
        monomers (list): A list of individual monomers.

    Raises:
        AssertionError: If the polymer does not start with "CHEM" or "RNA".

    """
    joint_monomers = polymer.split("{")[1].split("}")[0]
    monomers = joint_monomers.split(".")
    return monomers


def parse_monomer(monomer):
    """
    Parse a monomer string and extract the sugar, base, and phosphate components.

    Args:
        monomer (str): The monomer string to parse.

    Returns:
        sugar (str): The sugar component of the monomer.
        base (str): The base component of the monomer.
        phosphate (str): The phosphate component of the monomer.
    """
    if "(" not in monomer:
        # capture phosphate cap for siRNAs
        sugar = ""
        base = ""
        phosphate = monomer
    else:
        # split monomer into sugar, base, and phosphate
        sugar = monomer.split("(")[0]
        base = monomer.split("(")[1].split(")")[0]
        phosphate = monomer.split(")")[1]

    sugar = sugar.replace("[", "").replace("]", "")
    base = base.replace("[", "").replace("]", "")
    phosphate = phosphate.replace("[", "").replace("]", "")
    return sugar, base, phosphate


def xna2helm(xna: XNAData = None):
    """
    Convert an XNAData object to a HELM string representation.

    Args:
        xna (XNAData): The XNAData object to convert.

    Returns:
        helm (str): The HELM string representation of the XNAData object.
    """
    xna_polymers = dict(sorted(xna.polymers.items()))
    polymer_strings = []
    for polymer_name, polymer in xna_polymers.items():
        assert polymer_name.startswith(
            ("CHEM", "RNA")
        ), "Only CHEM and RNA polymers are supported"
        if polymer_name.startswith("CHEM"):
            chem_monomer = polymer["chemistry"]
            if len(chem_monomer) > 1:
                polymer_string = f"{polymer_name}{{[{chem_monomer}]}}"
            else:
                polymer_string = f"{polymer_name}{{{chem_monomer}}}"
        elif polymer_name.startswith("RNA"):
            sugar = polymer["sugar"].split(".")
            base = polymer["base"].split(".")
            phosphate = polymer["phosphate"].split(".")

            monomers = []
            for s, b, bb in zip(sugar, base, phosphate):
                if len(s) > 1:
                    s = f"[{s}]"
                if len(b) > 1:
                    b = f"([{b}])"
                if len(b) == 1:
                    b = f"({b})"
                if len(bb) > 1:
                    bb = f"[{bb}]"
                monomer = f"{s}{b}{bb}"
                monomers.append(monomer)

            monomer_string = ".".join(monomers)
            polymer_string = f"{polymer_name}{{{monomer_string}}}"
        polymer_strings.append(polymer_string)
    polymers_helm = "|".join(polymer_strings)

    if xna.connections is not None:
        connection_strings = []
        for connection in xna.connections:
            polymer_1 = connection["polymer_1"]
            polymer_2 = connection["polymer_2"]
            connection_type = connection["connection"]
            connection_string = f"{polymer_1},{polymer_2},{connection_type}"
            connection_strings.append(connection_string)
        connections_helm = "|".join(connection_strings)
    else:
        connections_helm = ""

    if xna.annotations is not None:
        annotations_helm = xna.annotations
    else:
        annotations_helm = ""

    helm = f"{polymers_helm}${connections_helm}$${annotations_helm}$"
    return helm