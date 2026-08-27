import unittest
import numpy as np
from src.depth.depth_calibration import DepthCalibration
from src.depth.depth_sampler import sample_bbox, sample_point


class DepthTests(unittest.TestCase):
    def test_robust_sampling(self):
        d=np.ones((20,20),dtype=float); d[10,10]=100; d[9,9]=np.nan
        self.assertEqual(sample_point(d,(10,10),7),1.0)
        self.assertEqual(sample_bbox(d,(4,4,16,16),.5),1.0)

    def test_linear_calibration(self):
        c=DepthCalibration(); c.fit([1,2,3],[2,4,6])
        self.assertAlmostEqual(c.correct(4),8,places=5); self.assertTrue(c.calibrated)


if __name__=="__main__": unittest.main()

