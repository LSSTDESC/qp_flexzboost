"""
Unit tests for PDF class
"""
import os

import numpy as np
import pytest
import qp
from qp import test_data
from qp.plotting import init_matplotlib
from qp.test_data import TEST_XVALS, XARRAY, XBINS, YARRAY
from qp.test_funcs import assert_all_close, assert_all_small, build_ensemble

from qp_flexzboost.flexzboost_pdf import FlexzboostGen


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

class TestEnsembleFunctions():
    """Test various ensemble related functions"""

    def test_built_ensemble(self, flexzboost_ensemble):
        """Ensure that the ensemble was properly built"""
        assert isinstance(flexzboost_ensemble.gen_obj, FlexzboostGen)

    @pytest.mark.skip(reason="There's a problem here currently")
    def test_can_create_pdf(self, flexzboost_ensemble, flexzboost_test_data):
        xpts = flexzboost_test_data['test_xvals']
        pdfs = flexzboost_ensemble.pdf(xpts)
        logpdfs = flexzboost_ensemble.logpdf(xpts)

        with np.errstate(all='ignore'):
            assert np.allclose(np.log(pdfs), logpdfs, atol=1e-9)

    @pytest.mark.skip(reason="There's a problem here currently")
    def test_can_create_cdf(self, flexzboost_ensemble, flexzboost_test_data):
        xpts = flexzboost_test_data['test_xvals']
        cdfs = flexzboost_ensemble.cdf(xpts)
        logcdfs = flexzboost_ensemble.logcdf(xpts)

        with np.errstate(all='ignore'):
            assert np.allclose(np.log(cdfs), logcdfs, atol=1e-9)

    def test_num_pdfs_match(self, flexzboost_ensemble):
        """Verify that number of pdfs in the ensemble match the generator object"""
        if hasattr(flexzboost_ensemble.gen_obj, 'npdf'):
            assert flexzboost_ensemble.npdf == flexzboost_ensemble.gen_obj.npdf

    @pytest.mark.skip(reason="There's a problem here currently")
    def test_can_convert_to_interp(self, flexzboost_ensemble):
        interp_ensemble = flexzboost_ensemble.convert_to(qp.interp_gen, np.linspace(0,3,100))

    #     binw = xpts[1:] - xpts[0:-1]
    #     check_cdf = ((pdfs[:,0:-1] + pdfs[:,1:]) * binw /2).cumsum(axis=1) - cdfs[:,1:]
    #     assert_all_small(check_cdf, atol=5e-2, test_name="cdf")

    #     hist = ens.histogramize(xpts)[1]
    #     hist_check = ens.frozen.histogramize(xpts)[1]
    #     assert_all_small(hist-hist_check, atol=1e-5, test_name="hist")

    #     ppfs = ens.ppf(test_data.QUANTS)
    #     check_ppf = ens.cdf(ppfs) - test_data.QUANTS
    #     assert_all_small(check_ppf, atol=2e-2, test_name="ppf")

    #     sfs = ens.sf(xpts)
    #     check_sf = sfs + cdfs
    #     assert_all_small(check_sf-1, atol=2e-2, test_name="sf")

    #     _ = ens.isf(test_data.QUANTS)
    #     check_isf = ens.cdf(ppfs) + test_data.QUANTS[::-1]
    #     assert_all_small(check_isf-1, atol=2e-2, test_name="isf")

    #     samples = ens.rvs(size=1000)
    #     assert samples.shape[0] == ens.frozen.npdf
    #     assert samples.shape[1] == 1000

    #     median = ens.median()
    #     mean = ens.mean()
    #     var = ens.var()
    #     std = ens.std()
    #     entropy = ens.entropy()

    #     _ = ens.stats()
    #     modes = ens.mode(xpts)

    #     assert median.size == ens.npdf
    #     assert mean.size == ens.npdf
    #     assert np.std(mean) > 1e-8
    #     assert var.size == ens.npdf
    #     assert std.size == ens.npdf
    #     assert entropy.size == ens.npdf
    #     assert modes.size == ens.npdf

    #     integral = ens.integrate(limits=(ens.gen_obj.a, ens.gen_obj.a))
    #     interval = ens.interval(0.05)

    #     assert integral.size == ens.npdf
    #     assert interval[0].size == ens.npdf

    #     for N in range(3):
    #         moment_partial = ens.moment_partial(N, limits=(test_data.XMIN, test_data.XMAX))
    #         calc_moment = qp.metrics.calculate_moment(ens, N, limits=(test_data.XMIN, test_data.XMAX))
    #         assert_all_close(moment_partial, calc_moment, rtol=5e-2, test_name="moment_partial_%i" % N)

    #         sps_moment = ens.moment(N)
    #         assert sps_moment.size == ens.npdf

    #     init_matplotlib()
    #     axes = ens.plot(xlim=(xpts[0], xpts[-1]))
    #     ens.plot_native(axes=axes)

    #     red_ens = ens[np.arange(5)]
    #     red_pdf = red_ens.pdf(xpts)

    #     check_red = red_pdf - pdfs[0:5]
    #     assert_all_small(check_red, atol=1e-5, test_name="red")

    #     if hasattr(ens.gen_obj, 'npdf'): # skip scipy norm
    #         commList = [None]
    #         try:
    #             import mpi4py.MPI
    #             commList.append(mpi4py.MPI.COMM_WORLD)
    #         except ImportError:
    #             pass
    #         for comm in commList:
    #             try:
    #                 group, fout = ens.initializeHdf5Write("testwrite.hdf5", ens.npdf, comm)
    #             except TypeError:
    #                 continue
    #             ens.writeHdf5Chunk(group, 0, ens.npdf)
    #             ens.finalizeHdf5Write(fout)
    #             readens = qp.read("testwrite.hdf5")
    #             assert readens.metadata().keys() == ens.metadata().keys()
    #             assert readens.objdata().keys() == ens.objdata().keys()
    #             os.remove("testwrite.hdf5")
