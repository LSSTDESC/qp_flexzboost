"""This implements a PDF sub-class specifically for FlexZBoost"""

from enum import Enum
from typing import List

import numpy as np
from flexcode.basis_functions import BasisCoefs
from qp import add_class
from qp.parameterizations.base import Pdf_rows_gen
from qp.plotting import get_axes_and_xlims, plot_pdf_on_axes
from qp.utils.array import CASE_FACTOR, CASE_PRODUCT, get_eval_case
from qp.utils.interpolation import interpolate_multi_x_y
from scipy.stats import rv_continuous


# pylint: disable=invalid-name
class BasisSystem(Enum):
    """This enumerates the various basis systems that FlexCode supports

    Parameters
    ----------
    Enum : enum
        This enum inherits from the Enum class.
    """

    cosine = 1
    Fourier = 2
    db4 = 3


# pylint: disable=too-many-arguments,too-many-instance-attributes
class FlexzboostGen(Pdf_rows_gen):
    """Distribution based on weighted basis functions output from FlexZBoost.

    Notes
    -----
    This class is meant primarily to be a compact storage mechanism for output
    from FlexCode.
    """

    # pylint: disable=protected-access

    name = "flexzboost"
    version = 0

    _support_mask = rv_continuous._support_mask

    def __init__(
        self,
        weights: List[List[float]],
        basis_system_enum_value: int,
        z_min: float,
        z_max: float,
        bump_threshold: float,
        sharpen_alpha: float,
        *args,
        **kwargs
    ):
        """This is the primary constructor for this `qp` generator.

        Parameters
        ----------
        weights : List[List[float]]
            A list of lists were each element is a floating point value. The weights
            represent the contribution of each basis function to the final PDF.
            The shape of `weights` should be N x b, where N = number of PDFs
            and b = number of basis functions.
        basis_system_enum_value : int
            The enum id to define the FlexCode basis system used to produce the results
        z_min : float
            The minimum z value considered when producing results with FlexCode
        z_max : float
            The maximum z value considered when producing results with FlexCode
        bump_threshold : float
            A parameter used by FlexCode to remove small bumps from the results
        sharpen_alpha : float
            A parameter used by FlexCode to sharpen peaks in the results

        Returns
        -------
        flexzboost_gen
            PDF generator for FlexZBoost distributions

        Notes
        -----
        The argument list of this constructor is admittedly rather long. This approach
        makes it easier to interface with the greater `qp` infrastructure. To ease the
        burden on the user, there is a classmethod that allows passing a single parameter,
        a `Flexcode:BasisCoefs` object, that is unpacked.

        See the method `FlexzboostGen:create_from_basis_coef_object`.
        """

        # kwargs['shape'] is used to by the parent class to define the total
        # number of PDFs stored in this generator object.
        kwargs["shape"] = np.asarray(weights).shape
        super().__init__(*args, **kwargs)

        self._weights = np.asarray(weights)
        self._basis_system_enum_value = basis_system_enum_value
        self._z_min = z_min
        self._z_max = z_max
        self._bump_threshold = None
        self._sharpen_alpha = None

        # These two assignments all the use of property.setter functions, which
        # encapsulate some type checking and will also update the parent class
        # metadata as needed.
        self.bump_threshold = bump_threshold
        self.sharpen_alpha = sharpen_alpha

        self._basis_coefficients = self._build_basis_coef_object()

        self._xvals = None
        self._yvals = None
        self._ycumul = None

        self._addmetadata("basis_system_enum_value", self._basis_system_enum_value)
        self._addmetadata("z_min", self._z_min)
        self._addmetadata("z_max", self._z_max)
        self._addmetadata("bump_threshold", self._bump_threshold)
        self._addmetadata("sharpen_alpha", self._sharpen_alpha)
        self._addobjdata("weights", self._weights)

    @property
    def basis_system_enum(self) -> BasisSystem:
        """Return the BasisSystem enum for this object.

        Returns
        -------
        BasisSystem
            The BasisSystem enum that defines the basis system used for these results.
        """
        return BasisSystem(self._basis_system_enum_value)

    @property
    def z_min(self) -> float:
        """Return the minimum z value used for the results stored in this object.

        Returns
        -------
        float
            Minimum z value used to predict these results
        """
        return self._z_min

    @property
    def z_max(self) -> float:
        """Return the maximum z value used for the results stored in this object.

        Returns
        -------
        float
            Maximum z value used to predict these results
        """
        return self._z_max

    @property
    def bump_threshold(self) -> float:
        """Return the bump threshold used for the results stored in this object.

        Returns
        -------
        float
            Bump threshold value used to predict these results
        """
        return self._bump_threshold

    @bump_threshold.setter
    def bump_threshold(self, new_bump_threshold):
        """This is a setter for bump threshold that allows users to modify
        the parameter on the fly without rerunning the model.

        The conditional logic is a byproduct of the way that scipy will pass
        values to the __init__ method when taking a slice of an ensemble containing
        this generator.

        `new_bump_threshold` can be passed in as a 0 dimensional
        numpy array. For floats this is fine, but the comparison logic in Flexcode
        breaks when a numpy 0 dimensional array (i.e. scalar) `None` value is passed in.
        To account for this, we explicitly assign `None` when we detect a
        None-like input.

        Parameters
        ----------
        new_bump_threshold : float
            The new bump threshold to use in the BasisCoefs object.
        """

        # We use the `==` comparison because Numpy will broadcast the contents
        # of new_bump_threshold appropriately.
        # pylint: disable-next=singleton-comparison
        if new_bump_threshold == None:
            self._bump_threshold = None
        else:
            self._bump_threshold = new_bump_threshold

        # _addmetadata updates the parent class, so that slices into an ensemble
        # will create new instances of this class with the correct values.
        self._addmetadata("bump_threshold", self._bump_threshold)
        self._update_basis_coef_object()

    @property
    def sharpen_alpha(self) -> float:
        """Return the sharpen alpha used for the results stored in this object.

        Returns
        -------
        float
            Sharpen alpha value used to predict these results
        """
        return self._sharpen_alpha

    @sharpen_alpha.setter
    def sharpen_alpha(self, new_sharpen_alpha):
        """This is a setter for sharpen alpha that allows users to modify
        the parameter on the fly without rerunning the model.

        The conditional logic is a byproduct of the way that scipy will pass
        values to the __init__ method when taking a slice of an ensemble containing
        this generator.

        `new_sharpen_alpha` can be passed in as a 0 dimensional
        numpy array. For floats this is fine, but the comparison logic in Flexcode
        breaks when a numpy 0 dimensional array (i.e. scalar) `None` value is passed in.
        To account for this, we explicitly assign `None` when we detect a
        None-like input.

        Parameters
        ----------
        new_sharpen_alpha : float
            The new sharpen parameter to use in the BasisCoefs object.
        """

        # We use the `==` comparison because Numpy will broadcast the contents
        # of new_sharpen_alpha appropriately.
        # pylint: disable-next=singleton-comparison
        if new_sharpen_alpha == None:
            self._sharpen_alpha = None
        else:
            self._sharpen_alpha = new_sharpen_alpha

        # _addmetadata updates the parent class, so that slices into an ensemble
        # will create new instances of this class with the correct values.
        self._addmetadata("sharpen_alpha", self._sharpen_alpha)
        self._update_basis_coef_object()

    @property
    def basis_coefficients(self) -> BasisCoefs:
        """Return the BasisCoef object that was used to instantiate this object.

        Returns
        -------
        BasisCoefs
            Object used to initialize the class instance
        """
        return self._basis_coefficients

    def _build_basis_coef_object(self):
        """Private method that builds and returns a `FlexCode:BasisCoefs` object
        from the constructor parameters.

        Returns
        -------
        BasisCoefs
            Object used to initialize the class instance
        """
        return BasisCoefs(
            coefs=None,
            basis_system=BasisSystem(self._basis_system_enum_value).name,
            z_min=self._z_min,
            z_max=self._z_max,
            bump_threshold=self._bump_threshold,
            sharpen_alpha=self._sharpen_alpha,
        )

    def _update_basis_coef_object(self):
        """Simple method to update the `BasisCoefs` object 'in place'."""
        self._basis_coefficients = self._build_basis_coef_object()

    def _calculate_yvals_if_needed(self, xvals: List[float]) -> None:
        """If self._yvals is None or the xvals have changed, reevaluate the y values.

        Parameters
        ----------
        xvals : List[float]
            The x-values to evaluate the basis function.
        """
        if self._yvals is None or xvals is not self._xvals:
            self._evaluate_basis_coefficients(xvals)

    def _evaluate_basis_coefficients(self, xvals: List[float]) -> None:
        """Assign the list of x values to self._xvals. Use that grid to evaluate
        the y_values of PDFs using the weights and parameters stored in
        self._basis_coefficients.

        Parameters
        ----------
        xvals : List[float]
            The x-values to evaluate the analytical PDFs

        Notes
        -----
        We'll maintain a copy of the x values in memory for this object, but it
        won't be stored to disk.

        FlexCode requires that the x values be reshaped, we'll do that in the call
        to `.evaluate`, but we won't keep the reshaped x values in memory.

        The `.evaluate` method expects the `BasisCoefs` object to contain the
        output weights. So we'll add the weights back to the object for evaluation,
        and then remove them when we've completed `evaluation`. We do this to ensure
        that the value of weights is not accidentally stored to disk twice. Once
        as `self._weights`, and once as `self._basis_coefficients.coefs`.

        If storage to disk wasn't a concern, then this wouldn't be a problem. Due
        to the way that Python maintains references to values in memory, assigning
        the same values to `self._weights` and `self._basis_coefficients.coefs`
        doesn't actually use 2x the memory.
        """
        self._xvals = xvals
        self._basis_coefficients.coefs = self._weights

        #! I think that `evaluate` will accept a 2d array of values where each row
        # is a PDF and each column is an x value) But for now, let's stick with
        # the simple case of one set of x values for all PDFs. Ultimately we
        # might need to do something different with the self._xvals.reshape(-1,1)

        self._yvals = self._basis_coefficients.evaluate(self._xvals.reshape(-1, 1))
        self._basis_coefficients.coefs = None

    def _compute_ycumul(self, xvals: List[float]) -> None:
        """Compute the cumulative values of y given an x grid

        Parameters
        ----------
        xvals : List[float]
            The x-values to evaluate the cumulative y value.
        """
        self._evaluate_basis_coefficients(xvals)

        # Do the magic to calculate cumulative values of y
        copy_shape = np.array(self._yvals.shape)
        self._ycumul = np.ndarray(copy_shape)
        self._ycumul[:, 0] = 0.5 * self._yvals[:, 0] * (self._xvals[1] - self._xvals[0])
        self._ycumul[:, 1:] = np.cumsum(
            (self._xvals[1:] - self._xvals[:-1]) * 0.5 * np.add(self._yvals[:, 1:], self._yvals[:, :-1]),
            axis=1,
        )

    # pylint: disable-next=arguments-differ
    def _pdf(self, x: List[float], row: List[int]) -> List[List[float]]:
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
        case_idx, xx, _ = get_eval_case(x, row)
        if case_idx in [CASE_PRODUCT, CASE_FACTOR]:
            self._calculate_yvals_if_needed(xx)
            return self._yvals.ravel()

        raise ValueError("Only CASE_PRODUCT and CASE_FACTOR are supported.")

    # pylint: disable-next=arguments-differ
    def _cdf(self, x: List[float], row: List[int]) -> List[List[float]]:
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
        case_idx, xx, _ = get_eval_case(x, row)
        # if self._ycumul is None:  # pragma: no cover
        self._compute_ycumul(xx)
        # if case_idx in [CASE_PRODUCT, CASE_FACTOR]:
        #     self._compute_ycumul(xx)
        # else:
        #     raise ValueError("Only CASE_PRODUCT and CASE_FACTOR are supported.")

        return self._ycumul.ravel()

    def _logcdf(self, x: List[float], row: List[int]) -> List[List[float]]:
        """Return the numerical log CDF, evaluated on the grid, `x`.

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

        return np.log(self._cdf(x, row))

    # pylint: disable-next=arguments-differ
    def _ppf(self, x: List[float], row: List[int]) -> List[List[float]]:
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

        self._xvals = np.linspace(self._z_min, self._z_max, 100)
        self._compute_ycumul(self._xvals)

        return interpolate_multi_x_y(
            x, row, self._ycumul, self._xvals, bounds_error=False, fill_value=(np.min(x), np.max(x))
        ).ravel()

    def _updated_ctor_param(self):
        """Specify the constructor parameters. This is required by scipy in order
        extend the rv_continuous class.

        Returns
        -------
        dct dict
            Dictionary of the constructor arguments and object instance variables
            needed to create this object.
        """
        dct = super()._updated_ctor_param()
        dct["weights"] = self._weights
        dct["basis_system_enum_value"] = self._basis_system_enum_value
        dct["z_min"] = self._z_min
        dct["z_max"] = self._z_max
        dct["bump_threshold"] = self._bump_threshold
        dct["sharpen_alpha"] = self._sharpen_alpha
        return dct

    @classmethod
    def create_from_basis_coef_object(
        cls, weights: List[List[float]], basis_coefficients_object: BasisCoefs, **kwargs
    ) -> Pdf_rows_gen:
        """This is a convenience method that allows the user to define a generator
        by passing a `BasisCoefs` object, instead of the typical 5 additional
        values.

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
        FlexzboostGen
            Returns an instance of this class. Note that FlexzboostGen is a subclass
            of Pdf_rows_gen, the return type defined in the method signature.
        """
        generator_object = cls(
            weights=weights,
            basis_system_enum_value=BasisSystem[basis_coefficients_object.basis_system].value,
            z_min=basis_coefficients_object.z_min,
            z_max=basis_coefficients_object.z_max,
            bump_threshold=basis_coefficients_object.bump_threshold,
            sharpen_alpha=basis_coefficients_object.sharpen_alpha,
        )

        return generator_object(**kwargs)

    @classmethod
    def get_allocation_kwds(cls, npdf, **kwargs):
        """Return the keywords necessary to create an 'empty' hdf5 file with npdf entries
        for iterative file write out. We only need to allocate the objdata columns, as
        the metadata can be written when we finalize the file.

        Parameters
        ----------
        npdf : int
            The total number of PDFs that will be written out

        Returns
        -------
        dict
            A dictionary that defines the storage requirements for this object.
        """
        try:
            weights = kwargs["weights"]
        except KeyError as key_error:
            raise KeyError("Required argument `weights` was not included in kwargs") from key_error

        num_weights = np.shape(weights)[-1]
        return {"weights", ((npdf, num_weights), "f4")}

    @classmethod
    def plot_native(cls, pdf, **kwargs):
        """Plot the PDF in a way that is particular to this type of distribution

        Here we'll use interpolated x,y points derived from the weights and FlexCode
        evaluation parameters.
        """
        axes, xlim, kwarg = get_axes_and_xlims(**kwargs)
        xvals = np.linspace(xlim[0], xlim[1], kwarg.pop("npts", 101))
        return plot_pdf_on_axes(axes, pdf, xvals, **kwarg)

    @classmethod
    def add_mappings(cls):
        """
        Add this classes mappings to the conversion dictionary
        """
        cls._add_creation_method(cls.create, None)
        cls._add_creation_method(cls.create_from_basis_coef_object, "basis_coef_object")

    @classmethod
    def make_test_data(cls):
        """Make data for unit tests"""
        WEIGHTS = np.asarray(
            [
                [
                    0.99999994,
                    1.4135911,
                    1.3578598,
                    1.3848811,
                    1.1752609,
                    1.2507105,
                    0.96589327,
                    1.2579455,
                    1.1328095,
                    0.9338199,
                    1.3668357,
                    0.63097477,
                    0.19285281,
                    -0.08388292,
                    0.05250954,
                    -0.5464654,
                    -0.3771514,
                    -0.3948611,
                    0.13923086,
                    -0.20495746,
                    -0.58977485,
                    -0.6391217,
                    -0.46343976,
                    -0.5011808,
                    -0.01433064,
                    0.278602,
                    0.5333237,
                    0.826034,
                    0.06464108,
                    0.9108775,
                    0.6811071,
                    0.69773537,
                    -0.11616451,
                    -0.09364327,
                    0.63583785,
                ],
                [
                    0.99999994,
                    1.3128049,
                    1.4268231,
                    1.3475941,
                    1.3009573,
                    1.1934606,
                    1.1979764,
                    1.4587557,
                    1.0695385,
                    1.0334687,
                    0.85049105,
                    0.6772867,
                    0.8599958,
                    0.7309471,
                    0.30866015,
                    0.10747848,
                    0.1454999,
                    0.4564285,
                    0.83178055,
                    0.9569013,
                    0.2805161,
                    0.35286552,
                    0.58561605,
                    0.42757383,
                    0.40403488,
                    -0.5502439,
                    0.56439424,
                    0.21782365,
                    0.80970615,
                    0.6189492,
                    0.9209366,
                    0.01046925,
                    -0.66917616,
                    0.0304801,
                    -0.34911576,
                ],
                [
                    0.99999994,
                    1.3046595,
                    1.3946912,
                    1.3725231,
                    1.3279371,
                    1.1379944,
                    1.1232849,
                    1.3168706,
                    1.1987064,
                    0.846475,
                    1.2190387,
                    1.0319941,
                    0.8385918,
                    0.72406054,
                    0.4407519,
                    1.0522529,
                    0.5317534,
                    0.82531404,
                    0.6055132,
                    0.42970878,
                    0.5682917,
                    0.42682788,
                    -0.04017492,
                    0.32071114,
                    0.7407263,
                    0.20112868,
                    0.28844437,
                    -0.01918357,
                    0.16105941,
                    -0.9992142,
                    -0.481242,
                    -0.3728989,
                    -0.39303133,
                    -0.556516,
                    -0.23944338,
                ],
                [
                    0.99999994,
                    1.405193,
                    1.3786027,
                    1.3832911,
                    1.3786896,
                    1.1868116,
                    1.1039548,
                    1.056342,
                    1.253356,
                    1.275163,
                    1.5149004,
                    0.7893624,
                    1.1212736,
                    0.7551946,
                    0.1665442,
                    0.31703034,
                    -0.3789813,
                    0.40208268,
                    -0.00154649,
                    -0.22578228,
                    -0.754486,
                    0.09544089,
                    -0.7406911,
                    -1.5187913,
                    -1.0511639,
                    -0.9208054,
                    -0.52502257,
                    -0.79425025,
                    0.11232897,
                    -0.5873992,
                    -0.00291769,
                    -1.2490546,
                    0.18622968,
                    -0.4166289,
                    -0.16232875,
                ],
                [
                    0.99999994,
                    1.32483,
                    1.2688403,
                    0.8508245,
                    1.4554728,
                    1.2448467,
                    0.852745,
                    0.8741474,
                    1.0841464,
                    0.7697048,
                    1.1911153,
                    0.51762104,
                    1.1319616,
                    1.3946458,
                    0.82583827,
                    0.21972111,
                    -0.16429716,
                    -0.08124515,
                    0.0241714,
                    -0.07269649,
                    0.04703106,
                    0.4027557,
                    -1.1216148,
                    -0.8540991,
                    -0.7413664,
                    -0.35533333,
                    -0.47791988,
                    -0.39957288,
                    0.1695733,
                    -0.46430817,
                    -0.07995562,
                    -1.0972134,
                    -0.61197704,
                    -1.1898835,
                    -0.75323683,
                ],
                [
                    0.99999994,
                    1.4217128,
                    1.4090639,
                    1.3527906,
                    1.2788762,
                    1.0873253,
                    1.0570015,
                    1.1381446,
                    0.73468673,
                    0.4902846,
                    0.11609144,
                    -0.43022275,
                    -0.33087614,
                    0.3467521,
                    0.14698188,
                    -0.79639876,
                    -0.7686687,
                    -1.0865113,
                    -1.0686133,
                    -1.0762304,
                    -0.9354039,
                    -0.79879427,
                    -0.24612567,
                    0.01798107,
                    -0.2094559,
                    0.24940334,
                    0.12473647,
                    0.10005763,
                    0.23591852,
                    0.33464774,
                    0.64543843,
                    0.24140209,
                    0.8614289,
                    0.10955815,
                    -0.09307325,
                ],
                [
                    0.99999994,
                    -0.60270864,
                    0.3777081,
                    1.0040071,
                    0.5319608,
                    1.1732529,
                    0.21736576,
                    1.0385551,
                    0.85155064,
                    0.8202011,
                    0.7389486,
                    0.69682765,
                    0.1181715,
                    0.13482217,
                    0.7518282,
                    0.8588988,
                    0.2753361,
                    0.10158755,
                    0.53366745,
                    0.5017293,
                    0.22024332,
                    0.8345108,
                    0.3317933,
                    0.5323848,
                    0.741613,
                    0.215265,
                    0.3551328,
                    0.44486073,
                    0.07836582,
                    0.00493836,
                    0.583493,
                    0.23795973,
                    0.10176475,
                    -0.08585434,
                    -0.47022513,
                ],
                [
                    0.99999994,
                    1.403194,
                    1.3613293,
                    1.2763977,
                    1.0978196,
                    1.0092797,
                    0.87263453,
                    0.63493866,
                    0.3737632,
                    -0.02474818,
                    0.12842114,
                    -0.31487998,
                    -0.18406785,
                    -0.42329717,
                    -0.8819336,
                    -0.887077,
                    -0.913117,
                    -1.1706294,
                    -1.1096691,
                    -0.46700883,
                    -0.7291215,
                    -0.20483486,
                    -0.57670075,
                    -0.5173913,
                    0.17409407,
                    -0.34383368,
                    0.11131766,
                    0.29361913,
                    0.22329482,
                    0.4090505,
                    0.50041765,
                    1.040421,
                    0.7399761,
                    1.3841617,
                    1.0754173,
                ],
                [
                    0.99999994,
                    0.5964216,
                    0.46396077,
                    1.2265164,
                    1.0870706,
                    1.1584536,
                    0.89783925,
                    0.7338294,
                    0.7884262,
                    0.41392878,
                    0.27348533,
                    0.60299355,
                    -0.09960458,
                    0.6036693,
                    -0.01055456,
                    0.32332683,
                    -0.63185304,
                    0.11284541,
                    -0.30345288,
                    -0.72329307,
                    -0.2737094,
                    0.03923929,
                    -0.26043436,
                    -0.5889996,
                    0.09375673,
                    -0.27470988,
                    -0.03649841,
                    0.1934136,
                    -0.41822934,
                    -0.38939086,
                    -0.2009153,
                    -0.1781136,
                    0.81968015,
                    0.5067288,
                    0.54687506,
                ],
                [
                    0.99999994,
                    1.3862562,
                    1.3533832,
                    1.3327965,
                    1.3019644,
                    1.3206618,
                    1.3192286,
                    0.97659546,
                    1.0163264,
                    1.0176893,
                    0.57915735,
                    0.7081749,
                    0.7332014,
                    0.5191775,
                    0.07479973,
                    0.13503157,
                    0.25693908,
                    -0.13746639,
                    -0.06378681,
                    -0.2937861,
                    -0.2938108,
                    0.03345898,
                    -0.45815086,
                    -0.45607626,
                    -0.91071063,
                    -0.7797466,
                    -0.5807737,
                    -0.34890455,
                    -0.60276383,
                    -0.49033943,
                    -0.81330174,
                    -0.4416928,
                    -0.88592136,
                    -0.7070263,
                    0.02908602,
                ],
            ]
        )
        Z_MIN = 0.0
        Z_MAX = 3.0
        BUMP_THRESHOLD = 0.1
        SHARPEN_ALPHA = 1.2
        X_VALS = np.linspace(Z_MIN, Z_MAX, 100)

        cls.test_data = {
            "gen_func": flexzboost,
            "ctor_data": {
                "weights": WEIGHTS,
                "basis_system_enum_value": BasisSystem.cosine.value,
                "z_min": Z_MIN,
                "z_max": Z_MAX,
                "bump_threshold": BUMP_THRESHOLD,
                "sharpen_alpha": SHARPEN_ALPHA,
            },
            "test_xvals": X_VALS,
            "weights": WEIGHTS,
        }


flexzboost = FlexzboostGen
flexzboost_create_from_basis_coef_object = FlexzboostGen.create_from_basis_coef_object

add_class(FlexzboostGen)
