import unittest, numpy as np
from src.geometry.pointing_ray import make_ray, point_to_ray_distance


class RayTests(unittest.TestCase):
    def test_ray_and_distance(self):
        origin,direction=make_ray([0,0,1],[1,0,1])
        self.assertTrue(np.allclose(origin,[1,0,1])); self.assertTrue(np.allclose(direction,[1,0,0]))
        distance,t=point_to_ray_distance([3,1,1],origin,direction)
        self.assertAlmostEqual(distance,1); self.assertAlmostEqual(t,2)

    def test_rejects_behind(self):
        distance,t=point_to_ray_distance([-1,0,0],[0,0,0],np.array([1,0,0]))
        self.assertEqual(distance,float("inf")); self.assertLess(t,0)


if __name__=="__main__": unittest.main()

