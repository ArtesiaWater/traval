from traval import BinaryClassifier


def test_bc():
    bc = BinaryClassifier(9, 1, 9, 1)
    return bc


def test_mcc():
    bc = test_bc()
    assert bc.mcc() == 0.8
    return


def test_tpr():
    bc = test_bc()
    tpr = bc.true_positive_rate
    assert tpr == 0.9
    return


def test_fpr():
    bc = test_bc()
    fpr = bc.false_positive_rate
    assert fpr == 0.1
    return


def test_tnr():
    bc = test_bc()
    tnr = bc.true_negative_rate
    assert tnr == 0.9
    return


def test_fnr():
    bc = test_bc()
    fnr = bc.false_negative_rate
    assert fnr == 0.1
    return


def test_add():
    bc = test_bc()
    bcsum = bc + bc
    assert bcsum.tp == 18
    assert bcsum.fp == 2
    assert bcsum.tn == 18
    assert bcsum.fn == 2
    return
