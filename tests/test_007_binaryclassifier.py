# ruff: noqa: D100 D103
from pandas import Series

from traval import BinaryClassifier


def get_bc():
    return BinaryClassifier(9, 1, 9, 1)


def test_all_stats():
    bc = get_bc()
    stats = bc.get_all_statistics()
    answer = {
        "tp": 9.0,
        "fp": 1.0,
        "fn": 1.0,
        "tn": 9.0,
        "sensitivity": 0.9,
        "tpr": 0.9,
        "fnr": 0.1,
        "specificity": 0.9,
        "tnr": 0.9,
        "fpr": 0.1,
        "ppv": 0.9,
        "npv": 0.9,
        "fdr": 0.1,
        "for": 0.1,
        "acc": 0.9,
        "prev": 0.5,
        "informedness": 0.8,
        "mcc": 0.8,
    }
    assert (stats == Series(answer)).all()


def test_add():
    bc = get_bc()
    bcsum = bc + bc
    assert bcsum.tp == 18
    assert bcsum.fp == 2
    assert bcsum.tn == 18
    assert bcsum.fn == 2
