from src.experimental_3d_pointing.core import AnchorDepthResult,AnchorTemporalFilter


def test_anchor_filter_updates_valid_values_and_rejects_jump():
    temporal=AnchorTemporalFilter(True,.3,.10,3)
    first,status=temporal.update(AnchorDepthResult(True,.80,.01,5,"VALID"))
    second,status2=temporal.update(AnchorDepthResult(True,.82,.01,5,"VALID"))
    jumped,status3=temporal.update(AnchorDepthResult(True,1.20,.01,5,"VALID"))
    assert first==.8 and abs(second-.806)<1e-9 and status2=="VALID"
    assert jumped is None and status3=="ANCHOR_DEPTH_JUMP"


def test_invalid_frames_are_not_hidden_by_previous_value():
    temporal=AnchorTemporalFilter(True,.3,.10,2)
    temporal.update(AnchorDepthResult(True,.8,.01,5,"VALID"))
    value,status=temporal.update(AnchorDepthResult(False,status="GLOBAL_DEPTH_UNSTABLE"))
    assert value is None and status=="GLOBAL_DEPTH_UNSTABLE"

