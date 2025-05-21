import numpy as np
from qp.core.ensemble import Ensemble

NBIN = 61
QUANTS = np.linspace(0.01, 0.99, NBIN)

def assert_all_close(arr, arr2, **kwds):
    """A slightly more informative version of asserting allclose"""
    test_name = kwds.pop("test_name", "test")
    if not np.allclose(arr, arr2, **kwds):  # pragma: no cover
        raise ValueError(
            "%s %.2e %.2e %s"
            % (test_name, (arr - arr2).min(), (arr - arr2).max(), kwds)
        )


def assert_all_small(arr, **kwds):
    """A slightly more informative version of asserting allclose"""
    test_name = kwds.pop("test_name", "test")
    if not np.allclose(arr, 0, **kwds):  # pragma: no cover
        raise ValueError("%s %.2e %.2e %s" % (test_name, arr.min(), arr.max(), kwds))


def build_ensemble(test_data):
    """Build an ensemble from test data in a class"""
    gen_func = test_data["gen_func"]
    ctor_data = test_data["ctor_data"]
    method = test_data.get("method", None)
    try:
        ens = Ensemble(gen_func, data=ctor_data, method=method)
        ancil = test_data.get("ancil")
        if ancil is not None:
            ens.set_ancil(ancil)
        return ens
    except Exception as exep:  # pragma: no cover
        print("Failed to make %s %s %s" % (gen_func, ctor_data, exep))
        raise ValueError from exep