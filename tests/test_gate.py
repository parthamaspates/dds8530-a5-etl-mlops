from ml.train import gate_decision


def test_first_model_only_needs_the_floor():
    assert gate_decision(0.90, None, 0.85) == "promoted"
    assert gate_decision(0.80, None, 0.85) == "rejected: below MIN_ROC_AUC"


def test_model_worse_than_champion_is_rejected_even_above_floor():
    assert gate_decision(0.95, 0.97, 0.85) == "rejected: worse than champion"


def test_equal_or_better_model_is_promoted():
    assert gate_decision(0.97, 0.97, 0.85) == "promoted"
    assert gate_decision(0.98, 0.97, 0.85) == "promoted"
