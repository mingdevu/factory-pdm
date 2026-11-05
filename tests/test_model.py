def test_weighted_lgbm_beats_chance(trained_bundle):
    _, metrics = trained_bundle
    weighted = metrics["failure_classifiers"]["lgbm_weighted"]
    base_rate = metrics["dataset"]["failure_rate"]
    # PR-AUC의 '무작위 기준선'은 양성 비율이다. 그보다 훨씬 높아야 학습이 된 것
    assert weighted["pr_auc"] > base_rate * 5
    assert 0.0 <= weighted["recall"] <= 1.0


def test_metrics_contain_confusion_matrix(trained_bundle):
    _, metrics = trained_bundle
    matrix = metrics["anomaly_detector"]["isolation_forest"]["confusion_matrix"]
    assert sum(matrix.values()) == metrics["dataset"]["test"]
