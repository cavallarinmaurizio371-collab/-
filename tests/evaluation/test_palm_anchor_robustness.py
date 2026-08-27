import numpy as np

from src.experimental_3d_pointing.core import palm_anchor_candidates


def test_v2_pools_palm_patches_without_relying_on_one_point():
    depth=np.full((80,120),.72,np.float32); pixels=np.zeros((21,3),float)
    indices=[0,5,9,13,17]
    for index,x in zip(indices,(10,30,50,70,90)): pixels[index,:2]=[x,30]
    depth[27:34,87:94]=1.4
    v1,v2=palm_anchor_candidates(depth,pixels,indices,7,3,.04)
    assert v1.valid and v2.valid
    assert abs(v2.median_m-.72)<1e-5
    assert v2.method=="V2_POOLED_ROBUST"

