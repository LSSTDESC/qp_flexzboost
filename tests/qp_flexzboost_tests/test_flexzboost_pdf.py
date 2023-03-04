import numpy as np
from flexcode.basis_functions import BasisCoefs

from qp_flexzboost.flexzboost_pdf import FlexzboostGen


def test_initialize_flexzboost_pdf():
    """Basic test to check instantiation of the FlexzboostGen object."""
    test_coefs = np.ones(10, float)
    test_basis_system = 'cosine'
    test_z_min = 0.0
    test_z_max = 3.0
    test_bump_threshold = 0.05
    test_sharpen_alpha = 1.2
    test_basis_coef = BasisCoefs(test_coefs, test_basis_system, test_z_min,
                                 test_z_max, test_bump_threshold, test_sharpen_alpha)

    test_fzb_pdf = FlexzboostGen(weights=test_coefs, basis_coefficients=test_basis_coef)

    assert test_fzb_pdf is not None
