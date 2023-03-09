import numpy as np
from flexcode.basis_functions import BasisCoefs

from qp_flexzboost.flexzboost_pdf import FlexzboostGen


def test_initialize_flexzboost_pdf_with_basis_coef_object():
    """Basic test to check instantiation of the FlexzboostGen object."""
    test_coefs = np.ones(10, float)
    test_basis_system = 'cosine'
    test_z_min = 0.0
    test_z_max = 3.0
    test_bump_threshold = 0.05
    test_sharpen_alpha = 1.2
    test_basis_coef = BasisCoefs(test_coefs, test_basis_system, test_z_min,
                                 test_z_max, test_bump_threshold, test_sharpen_alpha)

    test_fzb_pdf = FlexzboostGen.create_from_basis_coef_object(
        weights=test_coefs,
        basis_coefficients_object=test_basis_coef)

    assert test_fzb_pdf is not None

def test_initialize_flexzboost_pdf_with_basis_parameters():
    """Basic test to check instantiation of the FlexzboostGen object."""
    test_coefs = np.ones(10, float)
    test_basis_system_enum_value = 1
    test_z_min = 0.0
    test_z_max = 3.0
    test_bump_threshold = 0.05
    test_sharpen_alpha = 1.2

    test_fzb_pdf = FlexzboostGen(
        weights=test_coefs,
        basis_system_enum_value=test_basis_system_enum_value,
        z_min=test_z_min,
        z_max=test_z_max,
        bump_threshold=test_bump_threshold,
        sharpen_alpha=test_sharpen_alpha)

    assert test_fzb_pdf is not None
