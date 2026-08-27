import numpy as np

from src.experimental_3d_pointing.core import palm_anchor_candidates


def test_v2_rejects_sparse_background_depth_outliers():
    depth=np.full((80,120),.8,np.float32); pixels=np.zeros((21,3),float); indices=[0,5,9,13,17]
    for index,x in zip(indices,(10,30,50,70,90)): pixels[index,:2]=[x,30]
    depth[29,9]=3.; depth[31,51]=2.5; depth[28,71]=.1
    _,v2=palm_anchor_candidates(depth,pixels,indices,7,3,.04)
    assert v2.valid and abs(v2.median_m-.8)<1e-5

