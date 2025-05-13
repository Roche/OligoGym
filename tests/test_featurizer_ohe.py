import numpy as np
import pandas as pd
import pytest
from oligogym.features import OneHotEncoder

ASO_TEST = "RNA1{[lna](T)[sp].[lna](A)[sp].d(G)[sp].d(T)[sp].d(T)[sp].d(A)[sp].d(T)[sp].d(T)[sp].d(C)[sp].d(C)[sp].d(A)[sp].d(T)[sp].d(T)[sp].d(C)[sp].[lna](C)[sp].[lna](C)}$$$$"
SIRNA_TEST = "RNA1{m(G)[sp].m(C)[sp].m(G)p.m(G)p.[fl2r](U)p.m(G)p.[fl2r](C)p.[fl2r](C)p.[fl2r](G)p.m(G)p.m(C)p.m(G)p.m(G)p.m(A)p.m(G)p.m(C)p.m(A)[sp].m(A)[sp].m(A)}|RNA2{[vp].m(U)[sp].[fl2r](U)[sp].m(U)p.m(G)p.m(C)p.[fl2r](U)p.m(C)p.[fl2r](C)p.[fl2r](G)p.m(C)p.m(C)p.m(G)p.m(G)p.[fl2r](C)p.m(A)p.[fl2r](C)p.m(C)p.m(G)p.m(C)[sp].m(G)[sp].m(C)}$$$RNA1{StrandType:ss}|RNA2{StrandType:as}$"

ASO_TEST_OHE_SUGAR = np.array([[[1, 0],
        [1, 0],
        [0, 1],
        [0, 1],
        [0, 1],
        [0, 1],
        [0, 1],
        [0, 1],
        [0, 1],
        [0, 1],
        [0, 1],
        [0, 1],
        [0, 1],
        [0, 1],
        [1, 0],
        [1, 0]]])

ASO_TEST_OHE_BASE = np.array([[[1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [1, 0, 0, 0],
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [1, 0, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 0, 1],
        [0, 1, 0, 0],
        [1, 0, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 0, 1],
        [0, 0, 0, 1]]])

ASO_TEST_OHE_PHOSPHATE = np.array([[[1, 0],
        [1, 0],
        [1, 0],
        [1, 0],
        [1, 0],
        [1, 0],
        [1, 0],
        [1, 0],
        [1, 0],
        [1, 0],
        [1, 0],
        [1, 0],
        [1, 0],
        [1, 0],
        [1, 0],
        [0, 1]]])

ASO_TEST_OHE_COMPLETE = np.array([[[1, 0, 1, 0, 1, 0, 0, 0],
        [1, 0, 1, 0, 0, 1, 0, 0],
        [1, 0, 0, 1, 0, 0, 1, 0],
        [1, 0, 0, 1, 1, 0, 0, 0],
        [1, 0, 0, 1, 1, 0, 0, 0],
        [1, 0, 0, 1, 0, 1, 0, 0],
        [1, 0, 0, 1, 1, 0, 0, 0],
        [1, 0, 0, 1, 1, 0, 0, 0],
        [1, 0, 0, 1, 0, 0, 0, 1],
        [1, 0, 0, 1, 0, 0, 0, 1],
        [1, 0, 0, 1, 0, 1, 0, 0],
        [1, 0, 0, 1, 1, 0, 0, 0],
        [1, 0, 0, 1, 1, 0, 0, 0],
        [1, 0, 0, 1, 0, 0, 0, 1],
        [1, 0, 1, 0, 0, 0, 0, 1],
        [0, 1, 1, 0, 0, 0, 0, 1]]])

SIRNA_TEST_OHE_SUGAR = np.array([[[1, 0, 0],
        [1, 0, 0],
        [1, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 1, 0],
        [0, 1, 0],
        [1, 0, 0],
        [1, 0, 0],
        [1, 0, 0],
        [1, 0, 0],
        [1, 0, 0],
        [1, 0, 0],
        [1, 0, 0],
        [1, 0, 0],
        [1, 0, 0],
        [1, 0, 0],
        [0, 0, 1],
        [1, 0, 0],
        [0, 1, 0],
        [1, 0, 0],
        [1, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 1, 0],
        [1, 0, 0],
        [1, 0, 0],
        [1, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [1, 0, 0],
        [0, 1, 0],
        [1, 0, 0],
        [1, 0, 0],
        [1, 0, 0],
        [1, 0, 0],
        [1, 0, 0]]])

SIRNA_TEST_OHE_BASE = np.array([[[1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [1, 0, 0, 0, 0],
        [1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [1, 0, 0, 0, 0],
        [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [1, 0, 0, 0, 0],
        [1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0],
        [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [1, 0, 0, 0, 0],
        [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0]]])

SIRNA_TEST_OHE_PHOSPHATE = np.array([[[1, 0, 0, 0],
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 1, 0, 0],
        [0, 1, 0, 0],
        [0, 1, 0, 0],
        [0, 1, 0, 0],
        [0, 1, 0, 0],
        [0, 1, 0, 0],
        [0, 1, 0, 0],
        [0, 1, 0, 0],
        [0, 1, 0, 0],
        [0, 1, 0, 0],
        [0, 1, 0, 0],
        [0, 1, 0, 0],
        [0, 1, 0, 0],
        [1, 0, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0],
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 1, 0, 0],
        [0, 1, 0, 0],
        [0, 1, 0, 0],
        [0, 1, 0, 0],
        [0, 1, 0, 0],
        [0, 1, 0, 0],
        [0, 1, 0, 0],
        [0, 1, 0, 0],
        [0, 1, 0, 0],
        [0, 1, 0, 0],
        [0, 1, 0, 0],
        [0, 1, 0, 0],
        [0, 1, 0, 0],
        [0, 1, 0, 0],
        [0, 1, 0, 0],
        [1, 0, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 1, 0]]])

SIRNA_TEST_OHE_COMPLETE = np.array([[[1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
        [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
        [0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
        [0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
        [0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
        [0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
        [0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
        [0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
        [0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
        [0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
        [0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
        [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
        [0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
        [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
        [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
        [1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
        [0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
        [0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
        [0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
        [0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
        [0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
        [0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
        [0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
        [0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
        [0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
        [0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
        [0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
        [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
        [0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
        [0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
        [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
        [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0]]])
        
# OneHotEncoder tests
def ohe_sugar():
    featurizer = OneHotEncoder(encode_components=["sugar"])
    return featurizer


def ohe_base():
    featurizer = OneHotEncoder(encode_components=["base"])
    return featurizer


def ohe_phosphate():
    featurizer = OneHotEncoder(encode_components=["phosphate"])
    return featurizer


def ohe_complete():
    featurizer = OneHotEncoder(encode_components=["sugar", "base", "phosphate"])
    return featurizer

@pytest.mark.parametrize(
    "featurizer", [ohe_sugar(), ohe_base(), ohe_phosphate(), ohe_complete()]
)
def test_feature_extraction_aso(featurizer):
    features = featurizer.fit_transform([ASO_TEST])
    if featurizer.encode_components == ["sugar"]:
        assert np.allclose(features, ASO_TEST_OHE_SUGAR)
    elif featurizer.encode_components == ["base"]:
        assert np.allclose(features, ASO_TEST_OHE_BASE)
    elif featurizer.encode_components == ["phosphate"]:
        assert np.allclose(features, ASO_TEST_OHE_PHOSPHATE)
    elif featurizer.encode_components == ["sugar", "base", "phosphate"]:
        assert np.allclose(features, ASO_TEST_OHE_COMPLETE)

@pytest.mark.parametrize(
    "featurizer", [ohe_sugar(), ohe_base(), ohe_phosphate(), ohe_complete()]
)
def test_feature_extraction_sirna(featurizer):
    features = featurizer.fit_transform([SIRNA_TEST])
    if featurizer.encode_components == ["sugar"]:
        assert np.allclose(features, SIRNA_TEST_OHE_SUGAR)
    elif featurizer.encode_components == ["base"]:
        assert np.allclose(features, SIRNA_TEST_OHE_BASE)
    elif featurizer.encode_components == ["phosphate"]:
        assert np.allclose(features, SIRNA_TEST_OHE_PHOSPHATE)
    elif featurizer.encode_components == ["sugar", "base", "phosphate"]:
        assert np.allclose(features, SIRNA_TEST_OHE_COMPLETE)

#TODO: test transform method?
#TODO: test padding? 