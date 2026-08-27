import numpy as np

from evaluation.phase2b_metrics import landing_error


def test_physical_landing_error_is_euclidean_mm():
    assert landing_error(np.array([30.,40.]),(0.,0.))==50.
    assert landing_error(None,(0.,0.)) is None

