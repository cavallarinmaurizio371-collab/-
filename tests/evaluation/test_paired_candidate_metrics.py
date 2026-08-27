from evaluation.phase2b_metrics import paired_candidate_metrics


def test_paired_comparison_counts_wins_and_differences():
    rows=[{"B_error_mm":20.,"C_error_mm":10.},
          {"B_error_mm":5.,"C_error_mm":8.},
          {"B_error_mm":7.,"C_error_mm":7.},
          {"B_error_mm":None,"C_error_mm":1.}]
    result=paired_candidate_metrics(rows)
    assert result["paired_trials"]==3
    assert result["B_win_count"]==1 and result["C_win_count"]==1
    assert result["tie_count"]==1 and result["median_C_minus_B_mm"]==0.
