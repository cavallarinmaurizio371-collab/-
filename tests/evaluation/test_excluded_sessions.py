from evaluation.phase2b_metrics import session_is_excluded


def test_invalid_phase2a5_session_is_permanently_excluded():
    excluded=["session_20260826_190513_243892"]
    assert session_is_excluded("session_20260826_190513_243892",excluded)
    assert not session_is_excluded("session_20260826_191039_450446",excluded)

