"""
Unit tests for PDF class
"""
import os

import numpy as np
import pytest
import qp
from qp.plotting import init_matplotlib
from helpers import assert_all_close, assert_all_small, build_ensemble, QUANTS

from qp_flexzboost.flexzboost_pdf import FlexzboostGen

# pylint: disable=redefined-outer-name


@pytest.fixture
def flexzboost_test_data():
    """Pytest fixutre that returns class data for FlexzboostGen class

    Returns
    -------
    dict
        Class test data
    """
    FlexzboostGen.make_test_data()
    return FlexzboostGen.test_data


@pytest.fixture
def flexzboost_ensemble(flexzboost_test_data):
    """Pytest fixture that yields qp.ensemble composed of FlexzboostGen PDFs.

    Parameters
    ----------
    flexzboost_test_data : pytest.fixture
        Class level test data

    Yields
    ------
    qp.Ensemble
        Ensemble composed on FlexzboostGen pdf objects
    """
    yield build_ensemble(flexzboost_test_data)


class TestEnsembleFunctions:
    """Test various ensemble related functions"""

    def test_built_ensemble(self, flexzboost_ensemble):
        """Ensure that the ensemble was properly built"""
        assert isinstance(flexzboost_ensemble.gen_obj, FlexzboostGen)

    def test_can_create_pdf(self, flexzboost_ensemble, flexzboost_test_data):
        """Ensure that pdfs can be created for the ensemble."""
        xpts = flexzboost_test_data["test_xvals"]
        pdfs = flexzboost_ensemble.pdf(xpts)
        logpdfs = flexzboost_ensemble.logpdf(xpts)

        with np.errstate(all="ignore"):
            assert np.allclose(np.log(pdfs), logpdfs, atol=1e-9)

    def test_can_create_cdf(self, flexzboost_ensemble, flexzboost_test_data):
        """Ensure that CDFs can be created for the ensemble"""
        xpts = flexzboost_test_data["test_xvals"]
        cdfs = flexzboost_ensemble.cdf(xpts)
        logcdfs = flexzboost_ensemble.logcdf(xpts)

        with np.errstate(all="ignore"):
            assert np.allclose(np.log(cdfs), logcdfs, atol=1e-9)

    def test_num_pdfs_match(self, flexzboost_ensemble):
        """Verify that number of pdfs in the ensemble match the generator object"""
        if hasattr(flexzboost_ensemble.gen_obj, "npdf"):
            assert flexzboost_ensemble.npdf == flexzboost_ensemble.gen_obj.npdf

    def test_can_convert_to_interp(self, flexzboost_ensemble):
        """Ensure that the ensemble can be converted to another generator type"""
        interp_ensemble = flexzboost_ensemble.convert_to(qp.interp_gen, xvals=np.linspace(0, 3, 100))
        assert isinstance(interp_ensemble.dist, qp.interp_gen)

    def test_pdf_sum_matches_cdf(self, flexzboost_ensemble, flexzboost_test_data):
        """Ensure that output CDF makes sense relative to the PDF."""
        xpts = flexzboost_test_data["test_xvals"]
        pdfs = flexzboost_ensemble.pdf(xpts)
        cdfs = flexzboost_ensemble.cdf(xpts)

        binw = xpts[1:] - xpts[0:-1]
        check_cdf = ((pdfs[:, 0:-1] + pdfs[:, 1:]) * binw / 2).cumsum(axis=1) - cdfs[:, 1:]
        assert_all_small(check_cdf, atol=5e-2, test_name="cdf")

    def test_histogramization(self, flexzboost_ensemble, flexzboost_test_data):
        """Ensure that generating histograms in different ways produces the same
        results."""
        xpts = flexzboost_test_data["test_xvals"]
        hist = flexzboost_ensemble.histogramize(xpts)[1]
        hist_check = flexzboost_ensemble.frozen.histogramize(xpts)[1]
        assert_all_small(hist - hist_check, atol=1e-5, test_name="hist")

    def test_can_create_ppf(self, flexzboost_ensemble):
        """This test checks the output of the PPF function has the right shape"""
        ppfs = flexzboost_ensemble.ppf(QUANTS)
        assert flexzboost_ensemble.npdf == ppfs.shape[0]
        assert ppfs.shape[1] == QUANTS.shape[0]

    def test_survival_function(self, flexzboost_ensemble, flexzboost_test_data):
        """Ensure that the survival function works as expected"""
        xpts = flexzboost_test_data["test_xvals"]
        sfs = flexzboost_ensemble.sf(xpts)
        cdfs = flexzboost_ensemble.cdf(xpts)
        check_sf = sfs + cdfs
        assert_all_small(check_sf - 1, atol=2e-2, test_name="sf")

    def test_inverse_survival_function(self, flexzboost_ensemble):
        """Test that the ISF output has the right shape"""
        isf = flexzboost_ensemble.isf(QUANTS)
        assert flexzboost_ensemble.npdf == isf.shape[0]
        assert isf.shape[1] == QUANTS.shape[0]

    def test_random_variates_size(self, flexzboost_ensemble):
        """Demonstrate that random variates can be extracted from the distributions"""
        samples = flexzboost_ensemble.rvs(size=1000)
        assert samples.shape[0] == flexzboost_ensemble.frozen.npdf
        assert samples.shape[1] == 1000

    def test_basic_median(self, flexzboost_ensemble):
        """Ensure that median returns results of the expected size"""
        median = flexzboost_ensemble.median()
        assert median.size == flexzboost_ensemble.npdf

    @pytest.mark.skip(reason="Slow test, somewhat redundant")
    def test_basic_mean(self, flexzboost_ensemble):
        """This is a long running test. >45 seconds. Ensure that mean works."""
        mean = flexzboost_ensemble.mean()
        assert mean.size == flexzboost_ensemble.npdf
        assert np.std(mean) > 1e-8

    def test_basic_mode(self, flexzboost_ensemble, flexzboost_test_data):
        """Ensure that mode works as expected"""
        xpts = flexzboost_test_data["test_xvals"]
        modes = flexzboost_ensemble.mode(xpts)
        assert modes.size == flexzboost_ensemble.npdf

    def test_integrate(self, flexzboost_ensemble):
        """Check that integration returns results of the expected size"""
        # gen_obj.a = -inf, gen_obj.b = inf
        integral = flexzboost_ensemble.integrate(
            limits=(flexzboost_ensemble.gen_obj.a, flexzboost_ensemble.gen_obj.b)
        )
        assert integral.size == flexzboost_ensemble.npdf

    def test_interval(self, flexzboost_ensemble):
        """Test basic interval functionality"""
        interval = flexzboost_ensemble.interval(0.05)
        assert interval[0].size == flexzboost_ensemble.npdf

    def test_moment(self, flexzboost_ensemble, flexzboost_test_data):
        """Test basic functionality of partial moments works as expected"""
        xpts = flexzboost_test_data["test_xvals"]
        moment_partial = flexzboost_ensemble.moment_partial(0, limits=(min(xpts), max(xpts)))
        calc_moment = qp.metrics.calculate_moment(flexzboost_ensemble, 0, limits=(min(xpts), max(xpts)))
        assert_all_close(moment_partial, calc_moment, rtol=5e-2, test_name="moment_partial_0")

        sps_moment = flexzboost_ensemble.moment(0)
        assert sps_moment.size == flexzboost_ensemble.npdf

    def test_plot_native(self, flexzboost_ensemble, flexzboost_test_data):
        """Ensure that basic plotting works"""
        xpts = flexzboost_test_data["test_xvals"]
        init_matplotlib()
        axes = flexzboost_ensemble.plot(xlim=(xpts[0], xpts[-1]))
        flexzboost_ensemble.plot_native(axes=axes)

    def test_slicing_ensemble(self, flexzboost_ensemble, flexzboost_test_data):
        """Make sure that the ensemble can be sliced"""
        xpts = flexzboost_test_data["test_xvals"]

        red_ens = flexzboost_ensemble[np.arange(5)]
        red_pdf = red_ens.pdf(xpts)
        pdfs = flexzboost_ensemble.pdf(xpts)

        check_red = red_pdf - pdfs[0:5]
        assert_all_small(check_red, atol=1e-5, test_name="red")

    def test_recover_data_after_writing(self, flexzboost_ensemble):
        """Test for information loss after writing"""
        try:
            group, fout = flexzboost_ensemble.initializeHdf5Write("testwrite.hdf5", flexzboost_ensemble.npdf)
        except TypeError:
            pass
        flexzboost_ensemble.writeHdf5Chunk(group, 0, flexzboost_ensemble.npdf)
        flexzboost_ensemble.finalizeHdf5Write(fout)
        readens = qp.read("testwrite.hdf5")
        assert readens.metadata.keys() == flexzboost_ensemble.metadata.keys()
        assert readens.objdata.keys() == flexzboost_ensemble.objdata.keys()
        os.remove("testwrite.hdf5")
