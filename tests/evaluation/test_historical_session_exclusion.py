from evaluation.coarse_historical_replay import build_audit


def test_incomplete_and_invalid_historical_sessions_are_not_accepted():
    audit = build_audit()
    incomplete = next(item for item in audit["sessions"]
                      if item["session"] == "session_20260826_205359_827058")
    assert not incomplete["usable"]
    assert "INVALID_INCOMPLETE_SESSION.txt" in incomplete["markers"]
    assert not audit["exact_replay_possible"]
