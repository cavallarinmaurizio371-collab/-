import numpy as np

from src.experimental_3d_pointing.core import robust_anchor_depth


def test_anchor_uses_robust_median_and_reports_mad():
    depth=np.full((80,100),.70,np.float32); pixels=np.zeros((21,3),float)
    for index,point in zip((0,5,9,13,17),((10,10),(30,10),(50,10),(70,10),(90,10))):
        pixels[index,:2]=point
    depth[9:12,89:92]=1.4
    result=robust_anchor_depth(depth,pixels,[0,5,9,13,17],3,3,.04)
    assert result.valid and result.valid_samples==5
    assert np.isclose(result.median_m,.70) and np.isclose(result.mad_m,0)


def test_anchor_rejects_large_cross_landmark_mad():
    depth=np.ones((50,50),np.float32); pixels=np.zeros((21,3),float)
    values=(.5,.6,.7,.8,.9)
    for index,(point,value) in enumerate(zip(((5,5),(15,5),(25,5),(35,5),(45,5)),values)):
        pixels[[0,5,9,13,17][index],:2]=point; x,y=point; depth[y-1:y+2,x-1:x+2]=value
    assert robust_anchor_depth(depth,pixels,[0,5,9,13,17],3,3,.04).status=="GLOBAL_DEPTH_UNSTABLE"
