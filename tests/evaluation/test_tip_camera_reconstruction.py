import numpy as np

from src.experimental_3d_pointing.core import reconstruct_tip_camera


def test_tip_origin_combines_global_anchor_and_rotated_relative_offset():
    anchor=np.array([.1,.2,.7]); rotation=np.array([[0,-1,0],[1,0,0],[0,0,1.]])
    result=reconstruct_tip_camera(anchor,rotation,[.03,0,-.05],[0,0,0])
    assert np.allclose(result,[.1,.23,.65])
