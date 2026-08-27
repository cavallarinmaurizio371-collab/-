from types import SimpleNamespace

import cv2
import numpy as np


INTRINSICS=SimpleNamespace(fx=800.,fy=800.,cx=320.,cy=240.,distortion=[0,0,0,0,0],
                           mode="APPROXIMATE_INTRINSICS_FALLBACK")


def synthetic_hand():
    rng=np.random.default_rng(17); world=rng.normal(0,.025,(21,3))
    world[5]=[-.010,.025,.015]; world[6]=[-.006,.008,-.010]
    world[7]=[-.002,-.010,-.035]; world[8]=[.002,-.030,-.065]
    rvec=np.array([.07,-.11,.03]); tvec=np.array([.01,-.01,.70])
    matrix=np.array([[800,0,320],[0,800,240],[0,0,1]],float)
    image,_=cv2.projectPoints(world,rvec,tvec,matrix,np.zeros(5))
    rotation,_=cv2.Rodrigues(rvec)
    pixels=np.column_stack([image.reshape(-1,2),np.zeros(21)])
    return world,pixels,rotation


def phase2_config():
    return {"pnp":{"max_reprojection_rmse_px":25.},
            "axis":{"max_residual_m":.012,"minimum_linearity":.70},
            "anchor":{"landmark_index":9,"sample_landmark_indices":[0,5,9,13,17],
                      "patch_size":3,"minimum_valid_samples":3,"max_depth_mad_m":.04},
            "temporal":{"enabled":True,"ema_alpha":.35,"max_angle_jump_deg":20.}}
