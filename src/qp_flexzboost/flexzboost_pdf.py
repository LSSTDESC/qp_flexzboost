"""This implements a PDF sub-class specifically for FlexZBoost"""
from typing import List

from flexcode.basis_functions import BasisCoefs
from qp.pdf_gen import Pdf_rows_gen
from scipy.stats import rv_continuous


class FlexzboostGen(Pdf_rows_gen):
    """Distribution based on weighted basis functions output from FlexZBoost.

    Notes
    -----
    Some notes about what this is.
    """

    name = 'flexzboost'
    version = 0

    _support_mask = rv_continuous._support_mask

    def __init__ (self, basis_coefficients:BasisCoefs, *args, **kwargs):
        """_summary_

        Parameters
        ----------
        basis_coefficients : BasisCoefs
            An object that contains the FlexZBoost output weights as well as the
            parameters required to define the set of basis functions.

        Returns
        -------
        flexzboost_gen
            PDF generator for FlexZBoost distributions
        """
        self._basis_coefficients = basis_coefficients
        super().__init__()

    @property
    def basis_coefficients(self)->BasisCoefs:
        """Return the BasisCoef object that was used to instantiate this object.

        Returns
        -------
        BasisCoefs
            Object used to initialize the class instance
        """
        return self._basis_coefficients

    def _pdf(self, x:List[float], row:List[int]) -> List[List[float]]:
        """Return the numerical PDFs, evaluated on the grid, `x`.

        Parameters
        ----------
        x : List[float]
            The x-values to evaluate the analytical PDFs
        row : List[int], optional
            The indices for which numerical PDFs should be generated

        Returns
        -------
        List[List[float]]
            A list of lists corresponding to individual PDF's y-values. Each of
            the outer lists is a single PDF. The elements of the inner list are
            the resulting y-values corresponding to the input x-values.
        """
        return self._basis_coefficients.evaluate(x)

    def _cdf(self, x:List[float], row:List[int]) -> List[List[float]]:
        """Return the numerical CDF, evaluated on the grid, `x`.

        Parameters
        ----------
        x : List[float]
            The x-values to evaluate the analytical CDFs
        row : List[int], optional
            The indices for which numerical PDFs should be generated

        Returns
        -------
        List[List[float]]
            A list of lists corresponding to individual CDF's y-values. Each of
            the outer lists is a single PDF. The elements of the inner list are
            the resulting y-values corresponding to the input x-values.
        """
        return [[0]]

    @classmethod
    def get_allocation_kwds(cls, npdf, **kwargs):
        return super().get_allocation_kwds(npdf, **kwargs)
