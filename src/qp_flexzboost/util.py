
import numpy as np

CASE_PRODUCT = 0
CASE_FACTOR = 1
CASE_2D = 2
CASE_FLAT = 3

def get_eval_case(x, row):
    """ Figure out which of the various input formats scipy.stats has passed us

    Parameters
    ----------
    x : array_like
        Pdf x-vals
    row : array_like
        Pdf row indices

    Returns
    -------
    case : `int`
        The case code
    xx : array_like
        The x-values properly shaped
    rr : array_like
        The y-values, properly shaped

    Notes
    -----
    The cases are:

    CASE_FLAT : x, row have shapes (n) , (n) and do not factor
    CASE_FACTOR : x, row can be factors to shapes (1, nx) and (npdf, 1)
    CASE_PRODUCT : x, row have shapes (1, nx) and (npdf, 1)
    CASE_2D : x, row have shapes (npdf, nx) and (npdf, nx)
    """
    nd_x = np.ndim(x)
    nd_row = np.ndim(row)
    #if nd_x > 2 or nd_row > 2:  #pragma: no cover
    #    raise ValueError("Too many dimensions: x(%s), row(%s)" % (np.shape(x), np.shape(row)))
    if nd_x >= 2 and nd_row != 1:
        return CASE_2D, x, row
    if nd_x >= 2 and nd_row == 1:  #pragma: no cover
        raise ValueError("Dimension mismatch: x(%s), row(%s)" % (np.shape(x), np.shape(row)))
    if nd_row >= 2:
        return CASE_PRODUCT, x, row
    if np.size(x) == 1 or np.size(row) == 1:
        return CASE_FLAT, x, row
    xx = np.unique(x)
    rr = np.unique(row)
    if np.size(xx) == np.size(x):
        xx = x
    if np.size(rr) == np.size(row):
        rr = row
    if np.size(xx) * np.size(rr) != np.size(x):
        return CASE_FLAT, x, row
    return CASE_FACTOR, xx, np.expand_dims(rr, -1)