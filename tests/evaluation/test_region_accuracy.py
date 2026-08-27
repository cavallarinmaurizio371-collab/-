from evaluation.phase2b_metrics import region_accuracy


def test_region_accuracy_uses_only_actual_predictions():
    rows=[{"gt_target":"CENTER","B_region":"CENTER"},
          {"gt_target":"LEFT","B_region":"CENTER"},
          {"gt_target":"RIGHT","B_region":None}]
    assert region_accuracy(rows,"B")==.5

