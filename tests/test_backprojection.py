import unittest
from src.camera.calibration import CameraIntrinsics
from src.geometry.backprojection import pixel_depth_to_camera_xyz


class BackprojectionTests(unittest.TestCase):
    def test_principal_point_is_optical_axis(self):
        i=CameraIntrinsics(500,500,320,240,[0]*5,640,480)
        p=pixel_depth_to_camera_xyz(320,240,2,i)
        self.assertAlmostEqual(float(p[0]),0); self.assertAlmostEqual(float(p[1]),0); self.assertAlmostEqual(float(p[2]),2)

    def test_axes(self):
        i=CameraIntrinsics(100,100,50,50,[0]*5,100,100)
        p=pixel_depth_to_camera_xyz(60,70,1,i)
        self.assertAlmostEqual(float(p[0]),.1,places=5); self.assertAlmostEqual(float(p[1]),.2,places=5)


if __name__=="__main__": unittest.main()

