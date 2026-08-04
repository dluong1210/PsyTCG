"""Precision, Recall, and F1 guards."""

from evaluation.metrics import confusion_counts, evaluate_run, precision_recall_f1


def run_of(*decision_lists):
    return {
        f"u{i}": [(str(index), decision, float(decision)) for index, decision in enumerate(items)]
        for i, items in enumerate(decision_lists)
    }


def test_perfect_classification():
    metrics = evaluate_run(run_of([1], [0]), {"u0": 1, "u1": 0})
    assert metrics == {"precision": 1.0, "recall": 1.0, "f1": 1.0}


def test_false_positive_and_false_negative():
    run = run_of([0], [1])
    golden = {"u0": 1, "u1": 0}
    assert confusion_counts(run, golden) == {"tp": 0, "fp": 1, "fn": 1, "tn": 0}
    assert precision_recall_f1(run, golden) == (0.0, 0.0, 0.0)


def test_missing_golden_user_is_a_negative_prediction():
    metrics = evaluate_run({}, {"positive": 1, "control": 0})
    assert metrics == {"precision": 0.0, "recall": 0.0, "f1": 0.0}


def test_unlabelled_prediction_users_are_ignored():
    metrics = evaluate_run(run_of([1], [1]), {"u0": 1})
    assert metrics == {"precision": 1.0, "recall": 1.0, "f1": 1.0}
