import unittest
from src.geometry.target_selector import HysteresisSelector, select_target
from src.types import CupDetection


class SelectorTests(unittest.TestCase):
    def test_selects_nearest_2d_ray_and_none(self):
        cups=[CupDetection(1,(0,0,10,10),(100,5),.9),CupDetection(2,(0,0,10,10),(100,80),.9)]
        selected,score=select_target(cups,(0,0),(1,0),weight_2d=1,weight_3d=0,max_score=.5,max_2d_distance_px=100)
        self.assertEqual(selected,1)
        selected,_=select_target(cups,(0,0),(-1,0),max_2d_distance_px=100)
        self.assertIsNone(selected)

    def test_hysteresis(self):
        h=HysteresisSelector(3,2)
        self.assertIsNone(h.update(2)); self.assertIsNone(h.update(2)); self.assertEqual(h.update(2),2)
        self.assertEqual(h.update(None),2); self.assertIsNone(h.update(None))


if __name__=="__main__": unittest.main()

