import unittest
import numpy as np
from src.hand.gesture_classifier import classify_pointing


class GestureClassifierTests(unittest.TestCase):
    def test_extended_index_folded_other_fingers(self):
        p=np.zeros((21,3),dtype=float)
        p[0]=[0,0,0]; p[9]=[0,10,0]
        p[5]=[0,10,0]; p[6]=[0,20,0]; p[7]=[0,30,0]; p[8]=[0,40,0]
        for pip,tip,x in ((10,12,5),(14,16,8),(18,20,11)):
            p[pip]=[x,10,0]; p[tip]=[x/2,4,0]
        pointing,confidence=classify_pointing(p)
        self.assertTrue(pointing); self.assertGreaterEqual(confidence,.58)

    def test_missing_points(self):
        self.assertEqual(classify_pointing(np.zeros((3,3))), (False,0.0))


if __name__=="__main__": unittest.main()

