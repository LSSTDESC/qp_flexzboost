"""This implements a PDF sub-class specifically for FlexZBoost"""
from enum import Enum
from typing import List

import numpy as np
from flexcode.basis_functions import BasisCoefs
from qp.factory import add_class
from qp.pdf_gen import Pdf_rows_gen
from qp.plotting import get_axes_and_xlims, plot_pdf_on_axes
from qp.utils import interpolate_multi_x_y, interpolate_x_multi_y
from scipy.stats import rv_continuous


# pylint: disable=invalid-name
class BasisSystem(Enum):
    """_summary_

    Parameters
    ----------
    Enum : _type_
        _description_
    """
    cosine = 1
    Fourier = 2
    db4 = 3


# pylint: disable=too-many-arguments,too-many-instance-attributes
class FlexzboostGen(Pdf_rows_gen):
    """Distribution based on weighted basis functions output from FlexZBoost.

    Notes
    -----
    Some notes about what this is.
    """
    # pylint: disable=protected-access

    name = 'flexzboost'
    version = 0

    _support_mask = rv_continuous._support_mask

    def __init__(self, weights:List[List[float]], basis_system_enum_value:int,
                 z_min:float, z_max:float, bump_threshold:float,
                sharpen_alpha:float, *args, **kwargs):
        """_summary_

        Parameters
        ----------
        weights : List[List[float]]
            A list of lists were each element is a floating point value. The weights
            represent the contribution of each basis function to the final PDF.
            The shape of `weights` should be N x b, where N = number of PDFs
            and b = number of basis functions.

        basis_coefficients : BasisCoefs
            An object that contains the FlexZBoost output weights as well as the
            parameters required to define the set of basis functions.

        Returns
        -------
        flexzboost_gen
            PDF generator for FlexZBoost distributions
        """

        self._weights = np.asarray(weights)
        self._basis_system_enum_value = basis_system_enum_value
        self._z_min = z_min
        self._z_max = z_max
        self._bump_threshold = bump_threshold
        self._sharpen_alpha = sharpen_alpha

        self._basis_coefficients = self._build_basis_coef_object()

        self._xvals = None
        self._yvals = None
        self._ycumul = None

        super().__init__(*args, **kwargs)
        self._addmetadata('basis_system_enum_value', self._basis_system_enum_value)
        self._addmetadata('z_min', self._z_min)
        self._addmetadata('z_max', self._z_max)
        self._addmetadata('bump_threshold', self._bump_threshold)
        self._addmetadata('sharpen_alpha', self._sharpen_alpha)
        self._addobjdata('weights', self._weights)

    @property
    def basis_coefficients(self)->BasisCoefs:
        """Return the BasisCoef object that was used to instantiate this object.

        Returns
        -------
        BasisCoefs
            Object used to initialize the class instance
        """
        return self._basis_coefficients

    def _build_basis_coef_object(self):
        return BasisCoefs(coefs=None,
                          basis_system=BasisSystem(self._basis_system_enum_value).name,
                          z_min=self._z_min,
                          z_max=self._z_max,
                          bump_threshold=self._bump_threshold,
                          sharpen_alpha=self._sharpen_alpha)
    # def _clean_input_basis_coefficients(self, basis_coefficients) -> BasisCoefs:
    #     """This function will remove coefficients from the BasisCoef object to
    #     avoid duplicating storage.

    #     It will also convert back to a `BasisCoef` object if `qp` has converted
    #     it to a 0-dimensional numpy array.

    #     Parameters
    #     ----------
    #     basis_coefficients : BasisCoef (ideally
    #         The input object to be cleaned and type-checked.

    #     Returns
    #     -------
    #     BasisCoefs
    #         Cleaned version of the input.
    #     """

    #     returned_basis_coefficients = basis_coefficients

    #     # if qp machinery has converted this into a 0-dimensional array,
    #     # extract the original object
    #     if isinstance(basis_coefficients, np.ndarray):
    #         returned_basis_coefficients = np.expand_dims(basis_coefficients, 0)[0]

    #     # remove any coefs (i.e. weights) that are stored in the object.
    #     returned_basis_coefficients.coefs = None

    #     return returned_basis_coefficients

    def _calculate_yvals_if_needed(self, xvals:List[float]) -> None:
        """If self._yvals is None or the xvals have changed, reevaluate the y values.

        Parameters
        ----------
        xvals : List[float]
            The x-values to evaluate the basis function.
        """
        if self._yvals is None or xvals is not self._xvals:
            self._evaluate_basis_coefficients(xvals)

    def _evaluate_basis_coefficients(self, xvals:List[float]) -> None:
        """Assign the list of x values to self._xvals. Use that grid to evaluate
        the y_values of PDFs using the weights and parameters stored in
        self._basis_coefficients.

        Parameters
        ----------
        xvals : List[float]
            The x-values to evaluate the analytical PDFs
        """

        # ! Move these into these doc string as a notes section
        # We'll keep a copy of the x values in this object. Note - FlexCode
        # requires that the x values be reshaped, so we'll do that in the call
        # to `.evaluate`, but we won't keep the reshaped x values in the object.

        # The `.evaluate` method expects the `BasisCoefs` object to contain the
        # output weights. So we'll add the weights back to the object for evaluation,
        # and then remove them when we've completed `evaluation`.
        self._xvals = xvals
        self._basis_coefficients.coefs = self._weights
        self._yvals = self._basis_coefficients.evaluate(self._xvals.reshape(-1,1))
        self._basis_coefficients.coefs = None

    def _compute_ycumul(self, xvals:List[float]) -> None:
        """Compute the cumulative values of y given an x grid

        Parameters
        ----------
        xvals : List[float]
            The x-values to evaluate the cumulative y value
        """
        # Calculate yvals for the given xvals if needed
        self._calculate_yvals_if_needed(xvals)

        # Do the magic to calculate cumulative values of y
        copy_shape = np.array(self._yvals.shape)
        self._ycumul = np.ndarray(copy_shape)
        self._ycumul[:, 0] = 0.5 * self._yvals[:, 0] * (self._xvals[1] - self._xvals[0])
        self._ycumul[:, 1:] = np.cumsum((self._xvals[1:] - self._xvals[:-1]) *
                                        0.5 * np.add(self._yvals[:,1:],
                                                     self._yvals[:,:-1]), axis=1)

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
        # Calculate yvals for the given x's, if needed
        self._calculate_yvals_if_needed(x)

        return interpolate_x_multi_y(x, row, self._xvals, self._yvals,
                                     bounds_error=False, fill_value=0.).ravel()

    def _cdf(self, x:List[float], row:List[int]) -> List[List[float]]:
        """Return the numerical CDF, evaluated on the grid, `x`.

        Parameters
        ----------
        x : List[float]
            The x-values to evaluate the analytical CDFs
        row : List[int], optional
            The indices for which numerical CDFs should be generated

        Returns
        -------
        List[List[float]]
            A list of lists corresponding to individual CDF's y-values. Each of
            the outer lists is a single CDF. The elements of the inner list are
            the resulting y-values corresponding to the input x-values.
        """
        if self._ycumul is None:
            self._compute_ycumul(x)

        return interpolate_x_multi_y(x, row, self._xvals, self._ycumul,
                                     bounds_error=False, fill_value=(0.,1.)).ravel()

    def _ppf(self, x:List[float], row:List[int]) -> List[List[float]]:
        """Return the numerical PPF, evaluated on the grid, `x`.

        Parameters
        ----------
        x : List[float]
            The x-values to evaluate the analytical PPFs
        row : List[int], optional
            The indices for which numerical PPFs should be generated

        Returns
        -------
        List[List[float]]
            A list of lists corresponding to individual PPF's y-values. Each of
            the outer lists is a single PPF. The elements of the inner list are
            the resulting y-values corresponding to the input x-values.
        """
        if self._ycumul is None:  # pragma: no cover
            self._compute_ycumul(x)

        return interpolate_multi_x_y(x, row, self._ycumul, self._xvals,
            bounds_error=False, fill_value=(min(x), max(x))).ravel()

    def _updated_ctor_param(self):
        """
        Set weights and basis_coefficients as additional constructor argument
        """
        dct = super()._updated_ctor_param()
        dct['weights'] = self._weights
        dct['basis_system_enum_value'] = self._basis_system_enum_value
        dct['z_min'] = self._z_min
        dct['z_max'] = self._z_max
        dct['bump_threshold'] = self._bump_threshold
        dct['sharpen_alpha'] = self._sharpen_alpha
        return dct

    @classmethod
    def create_from_basis_coef_object(cls,
                                      weights:List[List[float]],
                                      basis_coefficients_object:BasisCoefs,
                                      **kwargs):
        """_summary_

        Parameters
        ----------
        weights : List[List[float]]
            _description_
        basis_coefficients_object : BasisCoefs
            _description_

        Returns
        -------
        _type_
            _description_
        """
        generator_object = cls(
            weights=weights,
            basis_system_enum_value=BasisSystem[basis_coefficients_object.basis_system].value,
            z_min=basis_coefficients_object.z_min,
            z_max=basis_coefficients_object.z_max,
            bump_threshold=basis_coefficients_object.bump_threshold,
            sharpen_alpha=basis_coefficients_object.sharpen_alpha)

        return generator_object(**kwargs)

    @classmethod
    def get_allocation_kwds(cls, npdf, **kwargs):
        """_summary_

        Parameters
        ----------
        npdf : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """
        return super().get_allocation_kwds(npdf, **kwargs)

    @classmethod
    def plot_native(cls, pdf, **kwargs):
        """Plot the PDF in a way that is particular to this type of distribution

        For a interpolated PDF this uses the interpolation points
        """
        axes, xlim, kwarg = get_axes_and_xlims(**kwargs)
        xvals = np.linspace(xlim[0], xlim[1], kwarg.pop('npts', 101))
        return plot_pdf_on_axes(axes, pdf, xvals, **kwarg)

    @classmethod
    def add_mappings(cls):
        """
        Add this classes mappings to the conversion dictionary
        """
        cls._add_creation_method(cls.create, None)
        cls._add_creation_method(cls.create_from_basis_coef_object, 'basis_coef_object')

    @classmethod
    def make_test_data(cls):
        """_summary_
        """


flexzboost = FlexzboostGen.create
flexzboost_create_from_basis_coef_object = FlexzboostGen.create_from_basis_coef_object

add_class(FlexzboostGen)
