import numpy as np
import pytest
from flexcode.basis_functions import BasisCoefs

from qp_flexzboost.flexzboost_pdf import BasisSystem, FlexzboostGen

# pylint: disable=redefined-outer-name

TEST_BASIS_SYSTEM = "cosine"
TEST_BASIS_SYSTEM_ENUM_VALUE = 1
TEST_Z_MIN = 0.0
TEST_Z_MAX = 3.0
TEST_BUMP_THRESHOLD = 0.05
TEST_SHARPEN_ALPHA = 1.2


@pytest.fixture
def flexzboost_ensemble():
    """Pytest fixture that yields a FlexZBoost generator object"""
    test_coefs = np.ones(10, float)
    test_basis_system = TEST_BASIS_SYSTEM
    test_z_min = TEST_Z_MIN
    test_z_max = TEST_Z_MAX
    test_bump_threshold = TEST_BUMP_THRESHOLD
    test_sharpen_alpha = TEST_SHARPEN_ALPHA
    test_basis_coef = BasisCoefs(
        test_coefs, test_basis_system, test_z_min, test_z_max, test_bump_threshold, test_sharpen_alpha
    )

    yield FlexzboostGen.create_from_basis_coef_object(
        weights=test_coefs, basis_coefficients_object=test_basis_coef
    )


def test_initialize_flexzboost_pdf_with_basis_coef_object(flexzboost_ensemble):
    """Basic test to check instantiation of the FlexzboostGen object."""
    assert flexzboost_ensemble is not None


def test_initialize_flexzboost_pdf_with_basis_parameters():
    """Basic test to check instantiation of the FlexzboostGen object."""
    test_coefs = np.ones(10, float)

    test_fzb_pdf = FlexzboostGen(
        weights=test_coefs,
        basis_system_enum_value=TEST_BASIS_SYSTEM_ENUM_VALUE,
        z_min=TEST_Z_MIN,
        z_max=TEST_Z_MAX,
        bump_threshold=TEST_BUMP_THRESHOLD,
        sharpen_alpha=TEST_SHARPEN_ALPHA,
    )

    assert test_fzb_pdf is not None


def test_properties(flexzboost_ensemble):
    """Basic test to ensure that getters return expected values"""
    assert flexzboost_ensemble.dist.z_min == TEST_Z_MIN
    assert flexzboost_ensemble.dist.z_max == TEST_Z_MAX
    assert flexzboost_ensemble.dist.bump_threshold == TEST_BUMP_THRESHOLD
    assert flexzboost_ensemble.dist.sharpen_alpha == TEST_SHARPEN_ALPHA
    assert flexzboost_ensemble.dist.basis_system_enum == BasisSystem(TEST_BASIS_SYSTEM_ENUM_VALUE)

    # This last assert will exercise the getting for basis_coefficients and then
    # pull z_min off of that object and compare it to the expected value.
    assert flexzboost_ensemble.dist.basis_coefficients.z_min == TEST_Z_MIN


def test_bump_threshold_setter_with_value(flexzboost_ensemble):
    """Basic test to exercise the bump_threshold setter for non-None values"""
    new_bump_threshold_value = 2.5
    flexzboost_ensemble.dist.bump_threshold = new_bump_threshold_value
    assert flexzboost_ensemble.dist.bump_threshold == new_bump_threshold_value


def test_bump_threshold_setter_with_none(flexzboost_ensemble):
    """Basic test to exercise the bump_threshold setter for a None value"""
    new_bump_threshold_value = None
    flexzboost_ensemble.dist.bump_threshold = new_bump_threshold_value
    assert flexzboost_ensemble.dist.bump_threshold == new_bump_threshold_value


def test_sharpen_alpha_setter_with_value(flexzboost_ensemble):
    """Basic test to exercise the sharpen_alpha setter for non-None values"""
    new_sharpen_alpha_value = 3.0
    flexzboost_ensemble.dist.sharpen_alpha = new_sharpen_alpha_value
    assert flexzboost_ensemble.dist.sharpen_alpha == new_sharpen_alpha_value


def test_sharpen_alpha_setter_with_none(flexzboost_ensemble):
    """Basic test to exercise the sharpen_alpha setter for None values"""
    new_sharpen_alpha_value = None
    flexzboost_ensemble.dist.sharpen_alpha = new_sharpen_alpha_value
    assert flexzboost_ensemble.dist.sharpen_alpha == new_sharpen_alpha_value


def test_allocation_kwds(flexzboost_ensemble):
    """Basic test to ensure that the allocation is correct"""
    kwargs = {"weights": [0.0, 1.0]}
    allocation_kwds = flexzboost_ensemble.dist.get_allocation_kwds(10, **kwargs)
    assert "weights" in allocation_kwds
    assert ((10, 2), "f4") in allocation_kwds


def test_allocation_kwds_missing_weights(flexzboost_ensemble):
    """Basic test to ensure that the allocation is correct"""
    kwargs = {}
    with pytest.raises(KeyError) as exception_info:
        _ = flexzboost_ensemble.dist.get_allocation_kwds(10, **kwargs)

    assert "Required argument `weights` was not" in str(exception_info.value)
