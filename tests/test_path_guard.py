import unittest
from src.safety.path_guard import PROJECT_ROOT, assert_safe_path


class PathGuardTests(unittest.TestCase):
    def test_accepts_project_child(self):
        self.assertEqual(assert_safe_path("outputs/a.txt"), (PROJECT_ROOT/"outputs/a.txt").resolve())

    def test_rejects_outside(self):
        with self.assertRaises(PermissionError): assert_safe_path(PROJECT_ROOT.parent/"outside.txt")


if __name__=="__main__": unittest.main()

