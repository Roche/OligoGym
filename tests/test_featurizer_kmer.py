import pytest
from oligogym.features import KMersCounts

ASO_TEST = "RNA1{[lna](T)[sp].[lna](A)[sp].d(G)[sp].d(T)[sp].d(T)[sp].d(A)[sp].d(T)[sp].d(T)[sp].d(C)[sp].d(C)[sp].d(A)[sp].d(T)[sp].d(T)[sp].d(C)[sp].[lna](C)[sp].[lna](C)}$$$$"
SIRNA_TEST = "RNA1{m(G)[sp].m(C)[sp].m(G)p.m(G)p.[fl2r](U)p.m(G)p.[fl2r](C)p.[fl2r](C)p.[fl2r](G)p.m(G)p.m(C)p.m(G)p.m(G)p.m(A)p.m(G)p.m(C)p.m(A)[sp].m(A)[sp].m(A)}|RNA2{[vp].m(U)[sp].[fl2r](U)[sp].m(U)p.m(G)p.m(C)p.[fl2r](U)p.m(C)p.[fl2r](C)p.[fl2r](G)p.m(C)p.m(C)p.m(G)p.m(G)p.[fl2r](C)p.m(A)p.[fl2r](C)p.m(C)p.m(G)p.m(C)[sp].m(G)[sp].m(C)}$$$RNA1{StrandType:ss}|RNA2{StrandType:as}$"

ASO_TEST_KMER_DICT = {
    'T': {0: 7},
    'A': {0: 3},
    'G': {0: 1},
    'C': {0: 5},
    'TA': {0: 2},
    'AG': {0: 1},
    'GT': {0: 1},
    'TT': {0: 3},
    'AT': {0: 2},
    'TC': {0: 2},
    'CC': {0: 3},
    'CA': {0: 1},
    'd': {0: 12},
    'lna': {0: 4},
    'sp': {0: 15},
    'EMPTY': {0: 1}
 }

SIRNA_TEST_KMER_MERGED_DICT = {
    'G': {0: 15},
    'C': {0: 15},
    'U': {0: 5},
    'A': {0: 5},
    'GC': {0: 9},
    'CG': {0: 7},
    'GG': {0: 4},
    'GU': {0: 1},
    'UG': {0: 2},
    'CC': {0: 4},
    'GA': {0: 1},
    'AG': {0: 1},
    'CA': {0: 2},
    'AA': {0: 2},
    'AU': {0: 1},
    'UU': {0: 2},
    'CU': {0: 1},
    'UC': {0: 1},
    'AC': {0: 1},
    'm': {0: 30},
    'fl2r': {0: 10},
    'EMPTY': {0: 3},
    'p': {0: 30},
    'sp': {0: 8},
    'vp': {0: 1}
}

SIRNA_TEST_KMER_SPLIT_DICT = {
    'RNA1_G': {0: 9},
    'RNA1_C': {0: 5},
    'RNA1_U': {0: 1},
    'RNA1_A': {0: 4},
    'RNA1_GC': {0: 4},
    'RNA1_CG': {0: 3},
    'RNA1_GG': {0: 3},
    'RNA1_GU': {0: 1},
    'RNA1_UG': {0: 1},
    'RNA1_CC': {0: 1},
    'RNA1_GA': {0: 1},
    'RNA1_AG': {0: 1},
    'RNA1_CA': {0: 1},
    'RNA1_AA': {0: 2},
    'RNA2_U': {0: 4},
    'RNA2_G': {0: 6},
    'RNA2_C': {0: 10},
    'RNA2_A': {0: 1},
    'RNA2_UU': {0: 2},
    'RNA2_UG': {0: 1},
    'RNA2_GC': {0: 5},
    'RNA2_CU': {0: 1},
    'RNA2_UC': {0: 1},
    'RNA2_CC': {0: 3},
    'RNA2_CG': {0: 4},
    'RNA2_GG': {0: 1},
    'RNA2_CA': {0: 1},
    'RNA2_AC': {0: 1},
    'RNA1_m': {0: 15},
    'RNA1_fl2r': {0: 4},
    'RNA1_p': {0: 14},
    'RNA1_sp': {0: 4},
    'RNA1_EMPTY': {0: 1},
    'RNA2_m': {0: 15},
    'RNA2_fl2r': {0: 6},
    'RNA2_EMPTY': {0: 2},
    'RNA2_p': {0: 16},
    'RNA2_sp': {0: 4},
    'RNA2_vp': {0: 1}
 }

def kmer_split():
    featurizer = KMersCounts(k=[1, 2], split_strands=True, modification_abundance=True)
    return featurizer


def kmer_merged():
    featurizer = KMersCounts(k=[1, 2], split_strands=False, modification_abundance=True)
    return featurizer


@pytest.mark.parametrize("featurizer", [kmer_merged()])
def test_feature_extraction_aso(featurizer):
    features = featurizer.fit_transform([ASO_TEST])
    assert features.to_dict() == ASO_TEST_KMER_DICT


@pytest.mark.parametrize("featurizer", [kmer_merged(), kmer_split()])
def test_feature_extraction_sirna(featurizer):
    features = featurizer.fit_transform([SIRNA_TEST])
    if featurizer.split_strands:
        assert features.to_dict() == SIRNA_TEST_KMER_SPLIT_DICT
    else:
        assert features.to_dict() == SIRNA_TEST_KMER_MERGED_DICT

#TODO: add transform test