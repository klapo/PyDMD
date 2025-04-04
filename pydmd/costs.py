"""
Module for the Coherent Spatio-Temporal Scale Separation with DMD.

References:
- Dylewsky, D., Tao, M., & Kutz, J. N. (2019). Dynamic mode decomposition for
multiscale nonlinear physics. Physics Review E, 99(6),
10.1103/PhysRevE.99.063311. https://doi.org/10.1103/PhysRevE.99.063311
"""

import copy

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import xarray as xr

from pydmd.bopdmd import BOPDMD
from .utils import compute_rank, compute_svd


class COSTS:
    """Coherent Spatio-Temporal Scale Separation with DMD.

    :param n_components: Number of independent frequency bands for this
        window length.
    :type n_components: int
    :param svd_rank: The rank of the BOPDMD fit.
    :type svd_rank: int
    :param global_svd: Flag indicating whether to find the proj_basis and
        initial values using the entire dataset instead of individually for
        each window. Generally using the global_svd speeds up the fitting
        process by not finding a new initial value for each window. Default
        is True.
    :type global_svd: bool
    :param initialize_artificially: Flag indicating whether to initialize the
        DMD using imaginary eigenvalues (i.e., the imaginary component of the
        cluster results from a previous iteration) through the
        `cluster_centroids` keyword. Default is False.
    :type initialize_artificially: bool
    :param pydmd_kwargs: Keyword arguments to pass onto the BOPDMD object.
    :type pydmd_kwargs: dict
    :param cluster_centroids: Cluster centroids from a previous fitting
        iteration to use for the initial guess of the eigenvalues. Should
        only be the imaginary component.
    :type cluster_centroids: numpy array
    :param reset_alpha_init: Flag indicating whether the initial guess for
        the BOPDMD eigenvalues should be reset for each window. Resetting the
        initial value increases the computation time due to finding a new
        initial guess. Default is False.
    :type reset_alpha_init: bool
    :param force_even_eigs: Flag indicating whether an even svd_rank should be
        forced when not specifying the svd_rank directly (i.e., svd_rank=0).
        Default is True.
    :type global_svd: bool
    :param max_rank: Maximum svd_rank allowed when the svd_rank is found
        through rank truncation (i.e., svd_rank=0).
    :type max_rank: int
    :param init_alpha: Initial guess for the eigenvalues provided to BOPDMD.
        Must be equal to the `svd_rank`.
    :type init_alpha: numpy array
    :param max_rank: Maximum allowed `svd_rank`. Overrides the optimal rank
        truncation if `svd_rank=0`.
    :type max_rank: int
    :param n_components: Number of frequency bands to use for clustering.
    :type n_components: int
    :param force_even_eigs: Flag specifying if the `svd_rank` should be forced
        to be even.
    :type force_even_eigs: bool
    :param reset_alpha_init: Flag specifying if the initial eigenvalue guess
        should be reset between windows.
    :type reset_alpha_init: bool
    :param kern_method: Specifies if the fit should be made to the data
    convolved with a kern that rounds towards zero at the edges of the time
    domain ("kern") or without a kerning ("flat").
    :type kern_method: string
    :param relative_filter_length: Value that determines how sharp the
    reconstruction convolution for each window is. Larger numbers mean that
    the convolution more heavily de-weights the edges of the time domain.
    Default is 2. Must be greater than zero.
    :type relative_filter_length: float
    """

    def __init__(
        self,
        svd_rank=None,
        global_svd=True,
        initialize_artificially=False,
        use_last_freq=False,
        init_alpha=None,
        pydmd_kwargs=None,
        cluster_centroids=None,
        reset_alpha_init=False,
        force_even_eigs=True,
        max_rank=None,
        n_components=None,
        kern_method=None,
        relative_filter_length=2,
    ):
        self._hist_kwargs = None
        self._omega_label = None
        self._step_size = None
        self._window_length = None
        self._n_components = n_components
        self._svd_rank = svd_rank
        self._global_svd = global_svd
        self._initialize_artificially = initialize_artificially
        self._use_last_freq = use_last_freq
        self._init_alpha = init_alpha
        self._cluster_centroids = cluster_centroids
        self._force_even_eigs = force_even_eigs
        self._max_rank = max_rank
        self._reset_alpha_init = reset_alpha_init
        self._relative_filter_length = relative_filter_length

        # Initialize variables that are defined in fitting.
        self._n_data_vars = None
        self._n_time_steps = None
        self._window_length = None
        self._n_slides = None
        self._time_array = None
        self._modes_array = None
        self._omega_array = None
        self._amplitudes_array = None
        self._cluster_centroids = None
        self._omega_classes = None
        self._transform_method = None
        self._window_means_array = None
        self._non_integer_n_slide = None
        self._svd_rank_pre_allocate = None

        # Specify how the data windows are constructed before fitting.
        if kern_method is None:
            self._kern_method = "kern"
        else:
            self._kern_method = kern_method

        # Specify default keywords to hand to BOPDMD.
        if pydmd_kwargs is None:
            self._pydmd_kwargs = {
                "eig_sort": "imag",
                "proj_basis": None,
                "use_proj": False,
            }
        else:
            self._pydmd_kwargs = pydmd_kwargs
            self._pydmd_kwargs["eig_sort"] = pydmd_kwargs.get(
                "eig_sort", "imag"
            )
            self._pydmd_kwargs["proj_basis"] = pydmd_kwargs.get(
                "proj_basis", None
            )
            self._pydmd_kwargs["use_proj"] = pydmd_kwargs.get("use_proj", False)

    @property
    def svd_rank(self):
        """Return the svd_rank used for the BOPDMD fit.

        :return: the rank used for the svd truncation.
        :rtype: int or float
        """
        return self._svd_rank

    @property
    def global_svd(self):
        """Return if a global svd projection basis was used.

        :return: If a global svd was used for the BOP-DMD fit.
        :rtype: int or float
        """
        return self._global_svd

    @property
    def window_length(self):
        """Return the window length used.

        :return: the length of the windows used for this decomposition level.
        :rtype: int or float
        """
        return self._window_length

    @property
    def step_size(self):
        """Return the step size between each window.

        :return: the length of the windows used for this decomposition level.
        :rtype: int or float
        """
        return self._step_size

    @property
    def n_slides(self):
        """Return the number of slides performed for each window.

        :return: number of window slides for this decomposition level.
        :rtype: int
        """
        return self._n_slides

    @property
    def modes_array(self):
        """Return the spatial modes of each window's fit.

        :return: Modes for each window
        :rtype: numpy.ndarray
        """
        return self._modes_array

    @property
    def amplitudes_array(self):
        """Return the amplitudes of each window's fit.

        :return: amplitudes of each window
        :rtype: numpy.ndarray
        """
        return self._amplitudes_array

    @property
    def omega_array(self):
        """Return the frequencies (omega) of each window's fit.

        :return: omega (a.k.a eigenvalues or time dynamics) for each window
        :rtype: numpy.ndarray
        """
        return self._omega_array

    @property
    def time_array(self):
        """Return the time values contained by each window.

        :return: time values for each fit window
        :rtype: numpy.ndarray
        """
        return self._time_array

    @property
    def window_means_array(self):
        """Return the array of window time means.

        :return: Time mean of the data in each window
        :rtype: numpy.ndarray
        """
        return self._window_means_array

    @property
    def n_components(self):
        """Return the number of frequency bands.

        :return: Number of frequency bands fit in the kmeans clustering
        :rtype: int
        """
        return self._n_components

    @property
    def cluster_centroids(self):
        """Return the frequency band centroids.

        :return: Centroids of the frequency bands
        :rtype: numpy.ndarray
        """
        return self._cluster_centroids

    @property
    def omega_classes(self):
        """Return the frequency band classifications.

        :return: Frequency band classifications, corresponds to omega_array
        :rtype: numpy.ndarray
        """
        return self._omega_classes

    @property
    def kern_method(self):
        """Return the kern method used for the reconstruction.

        :return: kern method used by the `build_kern` method
        :rtype: str
        """
        return self._kern_method

    @property
    def relative_filter_length(self):
        """Return the relative filter length used for the reconstruction.

        :return: The filter length for weighting the reconstruction of each
        window.
        :rtype: float
        """
        return self._relative_filter_length

    def periods(self):
        """Convert the omega array into periods.

        :return: Time dynamics converted to periods
        :rtype: numpy.ndarray
        """
        if self._omega_array is None:
            raise ValueError("The object must be fit first.")
        frequencies = np.abs(
            self._omega_array[self._omega_classes > 0].imag.flatten()
        )
        return 2 * np.pi / frequencies

    @staticmethod
    def relative_error(x_est, x_true):
        """Helper function for calculating the relative error.

        :param x_est: Estimated values (i.e. from reconstruction)
        :type x_est: numpy.ndarray
        :param x_true: True (or observed) values.
        :type x_true: numpy.ndarray
        :return: Relative error between observations and model.
        :rtype: numpy.ndarray
        """
        return np.linalg.norm(x_est - x_true) / np.linalg.norm(x_true)

    @staticmethod
    def _build_windows(data, window_length, step_size, integer_windows=False):
        """How many times integer slides fit the data for a given step and
        window size.

        :param data: 1D snapshots for fitting
        :type data: numpy.ndarray
        :param window_length: Length of the fitting window in units of time
            steps
        :type window_length: int
        :param step_size:  Distance to slide each window.
        :type step_size: int
        :param integer_windows: Whether to force an integer number of windows
        :type integer_windows: bool
        :return: Number of windows to fit.
        :rtype: int
        """
        if integer_windows:
            n_split = np.floor(data.shape[1] / window_length).astype(int)
        else:
            n_split = data.shape[1] / window_length

        n_steps = int((window_length * n_split))

        # Number of sliding-window iterations
        n_slides = np.floor((n_steps - window_length) / step_size).astype(int)

        return n_slides + 1

    @staticmethod
    def calculate_lv_kern(
        window_length, corner_sharpness=None, kern_method=None
    ):
        """Calculate the kerning window for suppressing real eigenvalues.

        :param corner_sharpness: Parameter specifying how sharp the kerning
            window should be. Default is 16.
        :type corner_sharpness: int
        :param window_length: Size of the window in time steps to kern.
        :type window_length: int
        :return: Kernel for convolving with the windowed data.
        :rtype: np.ndarray
        :param kern_method: Specify how the window should be built. "flat"
        means no weighting is applied and "kern" means a gaussian weighting is
        applied that is dictated by `corner_sharpness`. "forward" and
        "backward" are options used internally for fitting windows at the
        edges of the time domain.
        """

        if corner_sharpness is None:
            corner_sharpness = 16

        if kern_method == "kern":
            # This is the window kerning from the original implementation of
            # the sliding mrDMD algorithm. It rounds rather sharply the data
            # towards zero at the beginning and end of the window's time
            # domain. The intent of the window kerning was to suppress the
            # real components. However, this also has the effect of reducing
            # fit veracity as well as distorting derived time scales from the
            # imaginary eigenvalue components. This window kerning is no longer
            # the recommended practice.
            # Higher corner sharpness = sharper corners.
            lv_kern = (
                np.tanh(
                    corner_sharpness
                    * np.arange(0, window_length)
                    / window_length
                )
                - np.tanh(
                    corner_sharpness
                    * (np.arange(0, window_length) - window_length + 1)
                    / window_length
                )
                - 1
            )
        elif kern_method == "flat":
            # Do not apply a window kerning prior to fitting each window.
            lv_kern = np.ones(window_length)
        else:
            raise ValueError(
                f"Unrecognized argument for `kern_method` provided:"
                f" {kern_method}. Valid options are `flat` and `kern`."
            )

        return lv_kern

    @staticmethod
    def build_kern(window_length, relative_filter_length, direction=None):
        """Build the convolution kernel for the window reconstruction.

        Each window is convolved with a gaussian filter for the
        reconstruction, which weights points in the middle of the window and
        de-emphasizes the edges of the window that are more poorly fit.

        :param window_length: Length of the data window in units of time
        :type window_length: int
        :param relative_filter_length: A parameter governing how strongly
        weighted the windowed construction is. Larger values mean more
        strongly weighting the middle of the window.
        :type relative_filter_length: float
        :param direction: Specify the special cases for reconstructing
        windows at the beginning and end of the time domain.
        :type direction: string
        :return: Gaussian filter of length `window_length`
        :rtype: np.ndarray

        """
        recon_filter_sd = window_length / relative_filter_length
        recon_filter = np.exp(
            -((np.arange(window_length) - (window_length - 1) / 2) ** 2)
            / recon_filter_sd**2
        )
        # Do not apply the kerning at the end of the time domain. This
        # assists in fitting the last window of the decomposition and stops
        # the propagation of errors at the edge of the time domain.
        if direction == "forward":
            recon_filter[(window_length // 2) :] = 1
        # Do not apply the kerning at the beginning of the time domain. This
        # assists in fitting the last window of the decomposition and stops
        # the propagation of errors at the edge of the time domain.
        elif direction == "backward":
            recon_filter[: (window_length // 2)] = 1
        elif direction is not None:
            raise ValueError(
                f"Unrecognized option for `direction` provided. Provided "
                f"argument was {direction}. Valid options are `forward`, "
                f"`backward`, and `None`."
            )

        return recon_filter

    @staticmethod
    def _data_shape(data):
        """Returns the shape of the data.

        :param data: Data to fit with mrCOSTS.
        :type data: numpy.ndarray
        :return n_time_steps: Number of time steps.
        :rtype n_time_steps: int
        :return n_data_vars: Number of spatial variables.
        :rtype n_data_vars: int
        """
        n_time_steps = np.shape(data)[1]
        n_data_vars = np.shape(data)[0]
        return n_time_steps, n_data_vars

    def _build_proj_basis(self, data, svd_rank=None):
        """Build the projection basis.

        :param data: Data to fit with mrCOSTS.
        :type data: numpy.ndarray
        :param svd_rank: Rank to fit with COSTS.
        :type svd_rank: int
        :return: SVD projection basis for COSTS.
        :rtype: numpy.ndarray
        """
        self._svd_rank = compute_rank(data, svd_rank=svd_rank)
        # Recover the first r modes of the global svd
        return compute_svd(data, svd_rank=svd_rank)[0]

    def _build_initialization(self):
        """Method for making initial guess of DMD eigenvalues.

        :return: First guess of eigenvalues
        :rtype: numpy.ndarray or None
        """
        # User provided initial eigenvalues.
        if self._initialize_artificially and self._init_alpha is not None:
            return self._init_alpha
        # Initial eigenvalue guesses from kmeans clustering.
        elif (
            self._initialize_artificially
            and self._init_alpha is None
            and self._cluster_centroids is not None
        ):
            init_alpha = np.repeat(
                np.sqrt(self._cluster_centroids) * 1j,
                int(self._svd_rank / self._n_components),
            )
            init_alpha = init_alpha * np.tile(
                [1, -1], int(self._svd_rank / self._n_components)
            )
            return init_alpha
        # The user accidentally provided both methods of initializing the
        # eigenvalues.
        if (
            self._initialize_artificially
            and self._init_alpha is not None
            and self._cluster_centroids is not None
        ):
            raise ValueError(
                "Only one of `init_alpha` and `cluster_centroids` can be"
                " provided"
            )

        # In all other cases we return None and let the first iteration of
        # BOPDMD searches for the initial values.
        return None

    def fit(
        self,
        data,
        time,
        window_length,
        step_size,
        verbose=False,
        corner_sharpness=None,
    ):
        """Fit COherent SpatioTemporal Scale separation (COSTS).

        :param data: the input data to decompose (1D snapshots). Dimensions of
            space vs time.
        :type data: numpy.ndarray
        :param time: time series labeling the 1D snapshots
        :type time: numpy.ndarray
        :param window_length: decomposition window length in number of time
            steps.
        :type window_length: int
        :param step_size: Number of time steps to slide forward from the
            previous window.
        :type step_size: int
        :param verbose: notifies progress for fitting. Default is False.
        :type verbose: bool
        :param corner_sharpness: See `calculate_lv_kern`
        :type corner_sharpness: float or int
        """

        # Prepare window and data properties.
        self._window_length = window_length
        self._step_size = step_size
        self._n_time_steps, self._n_data_vars = self._data_shape(data)

        if not self._n_time_steps == time.size:
            raise ValueError("Data and time dimensions do not align.")

        self._n_slides = self._build_windows(
            data,
            self._window_length,
            self._step_size,
        )

        if self._window_length > self._n_time_steps:
            raise ValueError(
                (
                    f"Window length ({self._window_length}) is larger than the "
                    f"time dimension ({self._n_time_steps})"
                )
            )

        # If the window size and step size do not span the data in an integer
        # number of slides, we add one last window that has a smaller step
        # spacing relative to the other window spacings.
        n_slide_last_window = self._n_time_steps - (
            self._step_size * (self._n_slides - 1) + self._window_length
        )
        if n_slide_last_window > 0:
            self._n_slides += 1
            self._non_integer_n_slide = True
        else:
            self._non_integer_n_slide = False

        # Build the projection basis if using a global svd.
        if self._global_svd:
            u = self._build_proj_basis(data, svd_rank=self._svd_rank)
            self._pydmd_kwargs["proj_basis"] = u
            self._pydmd_kwargs["use_proj"] = self._pydmd_kwargs.get(
                "use_proj", False
            )
            self._svd_rank = compute_rank(data, svd_rank=self._svd_rank)
            self._svd_rank_pre_allocate = self._svd_rank
        elif not self._global_svd and self._svd_rank > 0:
            if self._force_even_eigs and self._svd_rank % 2:
                raise ValueError(
                    "svd_rank is odd, but force_even_eigs is True."
                )
            if self._svd_rank > self._n_data_vars:
                raise ValueError(
                    "Rank is larger than the data spatial dimension."
                )
            self._svd_rank_pre_allocate = compute_rank(
                data, svd_rank=self._svd_rank
            )
        # If not using a global svd or a specified svd_rank, local u from
        # each window is used instead. The optimal svd_rank may change when
        # using the locally optimal svd_rank. To deal with this situation in
        # the pre-allocation we give the maximally allowed svd_rank for
        # pre-allocation.
        elif self._max_rank is not None:
            self._svd_rank_pre_allocate = self._max_rank
        else:
            self._svd_rank_pre_allocate = self._n_data_vars

        # Pre-allocate all elements for the sliding window DMD.
        self._time_array = np.zeros((self._n_slides, self._window_length))
        self._modes_array = np.zeros(
            (self._n_slides, self._n_data_vars, self._svd_rank_pre_allocate),
            np.complex128,
        )
        self._omega_array = np.zeros(
            (self._n_slides, self._svd_rank_pre_allocate), np.complex128
        )
        self._amplitudes_array = np.zeros(
            (self._n_slides, self._svd_rank_pre_allocate), np.complex128
        )
        self._window_means_array = np.zeros((self._n_slides, self._n_data_vars))

        # Get initial values for the eigenvalues.
        self._init_alpha = self._build_initialization()

        # Initialize the BOPDMD object.
        optdmd = BOPDMD(
            svd_rank=self._svd_rank,
            init_alpha=self._init_alpha,
            **self._pydmd_kwargs,
        )

        # Round the corners of the windowed data towards zero, which shrinks
        # the real components of the fitted eigenvalues away from unrealistic
        # exponential growth.
        lv_kern = self.calculate_lv_kern(
            self._window_length,
            corner_sharpness=corner_sharpness,
            kern_method=self._kern_method,
        )

        # Perform the sliding window DMD fitting.
        for k in range(self._n_slides):
            if verbose and k % 50 == 0:
                print(f"{k:} of {self._n_slides:}")

            sample_slice = self.get_window_indices(k)
            data_window = data[:, sample_slice]
            original_time_window = time[:, sample_slice]

            # All windows are fit with the time array reset to start at t=0.
            t_start = original_time_window[:, 0]
            time_window = original_time_window - t_start

            # Subtract off the time mean before rounding corners.
            c = np.mean(data_window, 1, keepdims=True)
            data_window = data_window - c

            # Round the corners of the window.
            data_window = data_window * lv_kern

            # Reset optdmd between iterations
            if not self._global_svd:
                # Get the svd rank for this window. Uses rank truncation when
                # svd_rank is not fixed, i.e. svd_rank = 0, otherwise uses the
                # specified rank.
                _svd_rank = compute_rank(data_window, svd_rank=self._svd_rank)
                # Force svd rank to be even to allow for conjugate pairs.
                if self._force_even_eigs and _svd_rank % 2:
                    _svd_rank += 1
                # Force svd rank to not exceed a user specified amount.
                if self._max_rank is not None:
                    optdmd.svd_rank = min(_svd_rank, self._max_rank)
                else:
                    optdmd.svd_rank = _svd_rank
                optdmd.proj_basis = self._pydmd_kwargs["proj_basis"]

            # Fit the window using the optDMD.
            optdmd.fit(data_window, time_window)

            # Assign the results from this window.
            self._modes_array[k, :, : optdmd.modes.shape[-1]] = optdmd.modes
            self._omega_array[k, : optdmd.eigs.shape[0]] = optdmd.eigs
            self._amplitudes_array[k, : optdmd.eigs.shape[0]] = (
                optdmd.amplitudes
            )
            self._window_means_array[k] = c.flatten()
            self._time_array[k] = original_time_window

            # Reset optdmd between iterations
            if not self._global_svd:
                # The default behavior is to reset the optdmd object to use
                # the initial value from the first window.
                if not self._use_last_freq and not self._reset_alpha_init:
                    optdmd.init_alpha = self._init_alpha
                # Use the eigenvalues from this window to seed the next window.
                elif self._use_last_freq:
                    optdmd.init_alpha = optdmd.eigs
                # Remove the initial guess for the eigenvalues entirely. This
                # is much more computationally expensive.
                elif self._reset_alpha_init:
                    optdmd.init_alpha = None

    def get_window_indices(self, k):
        """Returns the window indices for slide `k`.

        Handles non-integer number of slides by making the last slide
        correspond to `slice(-window_length, None)`.

        :param k: Window to index
        :type k: int
        :return: slice indexing the given window
        :rtype: slice
        """
        # Get the window indices and data.
        sample_start = self._step_size * k
        if k == self._n_slides - 1 and self._non_integer_n_slide:
            return slice(-self._window_length, None)
        return slice(sample_start, sample_start + self._window_length)

    def cluster_omega(
        self,
        n_components,
        kmeans_kwargs=None,
        transform_method=None,
        method=MiniBatchKMeans,
    ):
        """Clusters fitted eigenvalues into frequency bands by the imaginary
        component.

        Assigns the clustering results to the object.

        :param n_components: Hyperparameter for k-means clustering, number of
            clusters.
        :type n_components: int
        :param kmeans_kwargs: Arguments for KMeans clustering. The default is
            random_state = 0.
        :type kmeans_kwargs: dict
        :param transform_method: How to transform omega. See docstring for
            valid options.
        :type transform_method: str or NoneType
        :param method: Clustering method following the sklearn pattern (has
            `fit_predict` and `n_clusters` keywords). Default is
            MiniBatchKMeans.
        :type method: method
        """

        cluster_centroids, omega_classes = self._cluster(
            n_components,
            kmeans_kwargs=kmeans_kwargs,
            transform_method=transform_method,
            method=method,
        )

        # Assign the results to the object.
        self._cluster_centroids = cluster_centroids
        self._omega_classes = omega_classes
        self._transform_method = transform_method
        self._n_components = n_components

    def _cluster(
        self,
        n_components,
        kmeans_kwargs=None,
        transform_method=None,
        method=MiniBatchKMeans,
    ):
        """Clusters fitted eigenvalues into frequency bands by the imaginary
        component.

        Helper function for clustering. Call `cluster_omega` instead.

        :param n_components: Hyperparameter for k-means clustering, number of
            clusters.
        :type n_components: int
        :param kmeans_kwargs: Arguments for KMeans clustering. The default is
            random_state = 0.
        :type kmeans_kwargs: dict or NoneType
        :param transform_method: How to transform omega. See docstring for
            valid options.
        :type transform_method: str or NoneType
        :param method: Clustering method following the sklearn pattern (has
            `fit_predict` and `n_clusters` keywords). Default is
            MiniBatchKMeans.
        :type method: method
        :return omega_classes: Classes defining the frequency bands ordered
            from the largest frequency to the smallest frequency.
        :rtype omega_classes: numpy.ndarray
        :return cluster_centroids: Centroids of the frequency bands. Order
            corresponds to the classes.
        :rtype cluster_centroids: numpy.ndarray
        """
        if kmeans_kwargs is None:
            kmeans_kwargs = {}
            random_state = 0
            kmeans_kwargs["random_state"] = kmeans_kwargs.get(
                "random_state", random_state
            )
        if not hasattr(method, "fit_predict") and callable(
            getattr(method, "fit_predict")
        ):
            raise ValueError(
                "Clustering method must have `fit_predict()` method."
            )
        clustering = method(n_clusters=n_components, **kmeans_kwargs)

        # Reshape the omega array into a 1d array
        omega_rshp = self.omega_array.reshape(
            self._n_slides * self._svd_rank_pre_allocate
        )
        omega_transform = self.transform_omega(
            omega_rshp, transform_method=transform_method
        )

        omega_classes = clustering.fit_predict(np.atleast_2d(omega_transform).T)
        omega_classes = omega_classes.reshape(
            self._n_slides, self._svd_rank_pre_allocate
        )
        cluster_centroids = clustering.cluster_centers_.flatten()

        # Sort the clusters by the centroid magnitude.
        idx = np.argsort(cluster_centroids)
        lut = np.zeros_like(idx)
        lut[idx] = np.arange(n_components)
        omega_classes = lut[omega_classes]
        cluster_centroids = cluster_centroids[idx]

        return cluster_centroids, omega_classes

    def transform_omega(self, omega_array, transform_method="absolute"):
        """Transform omega, primarily for clustering.
        Options for transforming omega are:
            "period": :math:`\\frac{1}{\\omega}`
            "log10": :math:`log10(\\omega)`
            "square_frequencies": :math:`\\omega^2`
            "absolute": :math:`|\\omega|`
        Default value is "absolute". All transformations and clustering are
        performed on the imaginary portion of omega.

        :param omega_array:
        :param transform_method:
        :return: transformed omega array
        :rtype: numpy.ndarray
        """
        # Apply a transformation to omega to improve frequency band separation
        if transform_method == "absolute":
            omega_transform = np.abs(omega_array.imag.astype("float"))
            self._omega_label = r"$|\omega|$"
            self._hist_kwargs = {"bins": 64}
        # Outstanding question: should this be the complex conjugate or
        # the imaginary component squared?
        elif transform_method == "square_frequencies":
            omega_transform = (omega_array.imag**2).real.astype("float")
            self._omega_label = r"$|\omega|^{2}$"
            self._hist_kwargs = {"bins": 64}
        elif transform_method == "log10":
            omega_transform = np.log10(np.abs(omega_array.imag.astype("float")))
            # Impute log10(0) with the smallest non-zero values in log10(omega).
            zero_imputer = omega_transform[np.isfinite(omega_transform)].min()
            omega_transform[~np.isfinite(omega_transform)] = zero_imputer
            self._omega_label = r"$log_{10}(|\omega|)$"
            self._hist_kwargs = {"bins": 64}
        elif transform_method == "period":
            omega_transform = 1 / np.abs(omega_array.imag.astype("float"))
            self._omega_label = "Period"
            self._hist_kwargs = {"bins": 64}
        else:
            raise ValueError(
                f"Transform method {transform_method:} not supported."
            )

        return omega_transform

    def cluster_hyperparameter_sweep(
        self,
        n_components_range=None,
        transform_method=None,
        method=MiniBatchKMeans,
        clustering_kwargs=None,
    ):
        """Hyperparameter search for number of frequency bands.

        Searches for the optimal number of clusters to use in kmeans clustering
        separation of the frequency bands. To best separate frequency bands
        it may be necessary to transform omega. Scores clusters using the
        silhouette score which can be slow to compute.

        Options for transforming omega are:
            "period": :math:`\\frac{1}{\\omega}`
            "log10": :math:`log10(\\omega)`
            "square_frequencies": :math:`\\omega^2`
            "absolute": :math:`|\\omega|`
        Default value is "absolute". All transformations and clustering are
        performed on the imaginary portion of omega.

        :param n_components_range: Range of n_components for the sweep.
        :type n_components_range: numpy.ndarray of ints
        :param transform_method: How to transform the imaginary component of
            omega.
        :type transform_method: str
        :param method: Clustering method following the sklearn pattern (has
            `fit_predict` and `n_clusters` keywords). Default is
            MiniBatchKMeans.
        :param clustering_kwargs: keywords to give to the clustering method.
        :type clustering_kwargs: dict
        :type method: method
        :return: optimal value of `n_components` for clustering.
        """
        if n_components_range is None:
            n_components_range = np.arange(
                np.max((self.svd_rank // 4, 2)),
                self.svd_rank // 2 + 1,
            )
        score = np.zeros_like(n_components_range, float)

        # Reshape the omega array into a 1d array. This is done here and not
        # in the _cluster() helper to reduce the number of times the variable
        # is computed.
        omega_rshp = self.omega_array.reshape(
            self._n_slides * self._svd_rank_pre_allocate
        )
        # Apply the transformation
        omega_transform = self.transform_omega(
            omega_rshp, transform_method=transform_method
        )

        for nind, n in enumerate(n_components_range):
            _, omega_classes = self._cluster(
                n_components=n,
                transform_method=transform_method,
                kmeans_kwargs=clustering_kwargs,
                method=method,
            )

            classes_reshape = omega_classes.reshape(
                self._n_slides * self._svd_rank_pre_allocate
            )

            score[nind] = silhouette_score(
                np.atleast_2d(omega_transform).T,
                np.atleast_2d(classes_reshape).T,
            )

        return n_components_range[np.argmax(score)]

    def plot_omega_histogram(self):
        """Histogram of fit frequencies.

        This plot is useful for assessing if the frequencies bands were well
        separated. A good choice of transformation and clustering will have
        clearly separated clusters.

        :return fig: Figure handle for the plot
        :return ax: Axes handle for the plot
        """

        # Apply the transformation to omega
        omega_transform = self.transform_omega(
            self.omega_array.flatten(), transform_method=self._transform_method
        )

        label = self._omega_label
        hist_kwargs = self._hist_kwargs

        cluster_centroids = self._cluster_centroids

        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        fig, ax = plt.subplots(1, 1)
        ax.hist(omega_transform, **hist_kwargs)
        ax.set_xlabel(label)
        ax.set_ylabel("Count")
        ax.set_title(r"$\omega$ Spectrum & k-Means Centroids")
        [
            ax.axvline(c, color=colors[nc % len(colors)])
            for nc, c in enumerate(cluster_centroids)
        ]

        return fig, ax

    def plot_omega_time_series(self):
        """Time series of transformed omega colored by frequency band.

        :return fig: figure handle for the plot
        :rtype fig: matplotlib.figure()
        :return ax: matplotlib subplot instances
        :rtype ax: matplotlib.Axes()
        """
        fig, ax = plt.subplots(1, 1)
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

        # Apply the transformation to omega
        omega_transform = self.transform_omega(
            self.omega_array.flatten(), transform_method=self._transform_method
        )

        label = self._omega_label

        for ncomponent, component in enumerate(range(self._n_components)):
            ax.plot(
                np.mean(self.time_array, axis=1),
                np.where(
                    self._omega_classes == component,
                    omega_transform.reshape(
                        (self._n_slides, self._svd_rank_pre_allocate)
                    ),
                    np.nan,
                ),
                color=colors[ncomponent % len(colors)],
                ls="None",
                marker=".",
            )
        ax.set_ylabel(label)
        ax.set_xlabel("Time")
        ax.set_title("Time dynamics time series")

        return fig, ax

    def global_reconstruction(self, scale_reconstruction_kwargs=None):
        """Helper function for generating the global reconstruction.

        :param scale_reconstruction_kwargs: Arguments for the scale
            reconstruction.
        :type scale_reconstruction_kwargs: dict
        :return: Global reconstruction (sum of all frequency bands)
        :rtype: numpy.ndarray
        """
        if scale_reconstruction_kwargs is None:
            scale_reconstruction_kwargs = {}
        xr_sep = self.scale_reconstruction(**scale_reconstruction_kwargs)
        x_global_recon = xr_sep.sum(axis=0)
        return x_global_recon

    def scale_reconstruction(
        self,
        include_means=True,
    ):
        """Reconstruct the spatiotemporal features for each frequency band.

        The reconstructed data are convolved with a guassian filter since
        points near the middle of the window are more reliable than points
        at the edge of the window. Note that this will leave the beginning
        and end of time series prone to larger errors. A best practice is
        to cut off `window_length` from each end before further analysis.

        :param include_means: Not API stable
        :return: Reconstruction for each frequency band with dimensions of:
            n_components x n_data_vars x n_time_steps
        :rtype: numpy.ndarray
        """

        # Each individual reconstructed window
        xr_sep = np.zeros(
            (self._n_components, self._n_data_vars, self._n_time_steps)
        )

        # Track the total contribution from all windows to each time step
        xn = np.zeros(self._n_time_steps)

        for k in range(self._n_slides):

            if k == 0:
                direction = "backward"
            elif k == self._n_slides - 1:
                direction = "forward"
            else:
                direction = None

            # Convolve each windowed reconstruction with a gaussian filter.
            # Weights points in the middle of the window and de-emphasizes the
            # edges of the window.
            recon_filter = self.build_kern(
                self._window_length,
                relative_filter_length=self._relative_filter_length,
                direction=direction,
            )

            window_indices = self.get_window_indices(k)

            w = self._modes_array[k]
            b = self._amplitudes_array[k]
            omega = copy.deepcopy(np.atleast_2d(self._omega_array[k]).T)
            classification = self._omega_classes[k]

            c = np.atleast_2d(self._window_means_array[k]).T

            # Compute each segment of the reconstructed data starting at "t = 0"
            t = self._time_array[k]
            t_start = t.min()
            t = t - t_start

            xr_sep_window = np.zeros(
                (self._n_components, self._n_data_vars, self._window_length)
            )
            for j in np.unique(self._omega_classes):
                class_index = classification == j
                xr_sep_window[j] = np.linalg.multi_dot(
                    [
                        w[:, class_index],
                        np.diag(b[class_index]),
                        np.exp(omega[class_index] * t),
                    ]
                ).real

                # Add the constant offset to the lowest frequency cluster.
                if include_means and j == np.argmin(self._cluster_centroids):
                    xr_sep_window[j] += c
                xr_sep_window[j] = xr_sep_window[j] * recon_filter

                xr_sep[j, :, window_indices] = (
                    xr_sep[j, :, window_indices] + xr_sep_window[j]
                )

            xn[window_indices] += recon_filter

        xr_sep = xr_sep / xn

        return xr_sep

    def scale_separation(
        self,
        scale_reconstruction_kwargs=None,
    ):
        """Separate the lowest frequency band from the high frequency bands.

        The lowest frequency band should contain the window means and can be
        passed on as the data for the next decomposition level. The high
        frequencies should have frequencies shorter than 1 / window_length.

        :param scale_reconstruction_kwargs: Arguments passed to
            `scale_reconstruction`
        :return xr_low_frequency: Reconstruction of the low frequency
            component with dimensions of n_data_vars x n_time_steps.
        :rtype xr_low_frequency: numpy.ndarray
        :return xr_high_frequency: Sum of all high frequency bands with
            dimensions of n_data_vars x n_time_steps
        :rtype xr_high_frequency: numpy.ndarray
        """

        if scale_reconstruction_kwargs is None:
            scale_reconstruction_kwargs = {}

        xr_sep = self.scale_reconstruction(**scale_reconstruction_kwargs)
        xr_low_frequency = xr_sep[0]
        xr_high_frequency = xr_sep[1:].sum(axis=0)

        return xr_low_frequency, xr_high_frequency

    def plot_scale_separation(
        self,
        data,
        scale_reconstruction_kwargs=None,
        plot_residual=False,
        fig_kwargs=None,
        plot_kwargs=None,
        hf_plot_kwargs=None,
        plot_contours=False,
    ):
        """Plot the scale-separated low and high frequency bands.

        The reconstructions are plotted in a time-space diagram. The high
        frequency component is the sum of all high frequency bands except the
        low frequency band which is plotted separately.

        :param data: Data used for the decomposition. An array of 1D snapshots.
        :type data: numpy.ndarray
        :param scale_reconstruction_kwargs: Arguments for reconstructing the
            COSTS fit.
        :type scale_reconstruction_kwargs: dict
        :param plot_residual: If the error should be fit. Will plot
            `data - low frequency - high frequency` yielding the error in
            absolute units.
        :type plot_residual: bool
        :param fig_kwargs: Arguments for the figure creation.
        :type fig_kwargs: dict
        :param plot_kwargs: Arguments for plotting the low frequency and data.
        :type plot_kwargs: dict
        :param hf_plot_kwargs: Arguments for the high frequency plotting.
        :type hf_plot_kwargs: dict
        :param plot_contours: Indicates if contours should be plotted.
        :type plot_contours: bool
        :return fig: figure handle for the plot
        :rtype fig: matplotlib.figure()
        :return axes: matplotlib subplot instances
        :rtype fig: matplotlib.Axes()
        """
        if scale_reconstruction_kwargs is None:
            scale_reconstruction_kwargs = {}

        xr_low_frequency, xr_high_frequency = self.scale_separation(
            scale_reconstruction_kwargs
        )

        if fig_kwargs is None:
            fig_kwargs = {}
        fig_kwargs["sharex"] = fig_kwargs.get("sharex", True)
        fig_kwargs["figsize"] = fig_kwargs.get("figsize", (6, 4))

        if plot_kwargs is None:
            plot_kwargs = {}
        plot_kwargs["vmin"] = plot_kwargs.get("vmin", -np.abs(data).max())
        plot_kwargs["vmax"] = plot_kwargs.get("vmax", np.abs(data).max())
        plot_kwargs["cmap"] = plot_kwargs.get("cmap", "cividis")

        if hf_plot_kwargs is None:
            hf_plot_kwargs = {}
        hf_plot_kwargs["vmin"] = hf_plot_kwargs.get(
            "vmin", -np.abs(xr_high_frequency).max()
        )
        hf_plot_kwargs["vmax"] = hf_plot_kwargs.get(
            "vmax", np.abs(xr_high_frequency).max()
        )
        hf_plot_kwargs["cmap"] = hf_plot_kwargs.get("cmap", "RdBu_r")

        if plot_residual:
            fig, axes = plt.subplots(4, 1, **fig_kwargs)
        else:
            fig, axes = plt.subplots(3, 1, **fig_kwargs)

        ax = axes[0]
        ax.pcolormesh(data, **plot_kwargs)
        if plot_contours:
            ax.contour(data, colors=["k"])
        ax.set_title(
            (
                f"Input data at decomposition window length ="
                f" {self._window_length:}"
            )
        )
        ax.set_ylabel("Space (-)")

        ax = axes[1]
        ax.set_title("Reconstruction, low frequency")
        ax.pcolormesh(xr_low_frequency, **plot_kwargs)
        if plot_contours:
            ax.contour(data, colors=["k"])
        ax.set_ylabel("Space (-)")

        ax = axes[2]
        ax.set_title("Reconstruction, high frequency")
        ax.pcolormesh(xr_high_frequency, **hf_plot_kwargs)
        ax.set_ylabel("Space (-)")

        if plot_residual:
            ax = axes[3]
            ax.set_title("Residual")
            ax.pcolormesh(
                data - xr_high_frequency - xr_low_frequency, **hf_plot_kwargs
            )
            ax.set_ylabel("Space (-)")

        axes[-1].set_xlabel("Time (-)")
        fig.tight_layout()

        return fig, axes

    def plot_reconstructions(
        self,
        data,
        plot_period=False,
        scale_reconstruction_kwargs=None,
        plot_residual=False,
        fig_kwargs=None,
        plot_kwargs=None,
        hf_plot_kwargs=None,
        plot_contours=False,
    ):
        """Time-space plots for each individual frequency band and the fitted
        data.

        :param data: Data used for the decomposition. An array of 1D snapshots.
        :type data: numpy.ndarray
        :param plot_period:
        :param scale_reconstruction_kwargs: Arguments for reconstructing the
            COSTS fit.
        :type scale_reconstruction_kwargs: dict
        :param plot_residual: Indicates if the residual of the fit should be
            plotted
        :type plot_residual: bool
        :param fig_kwargs: Arguments for the figure creation.
        :type fig_kwargs: dict
        :param plot_kwargs: Arguments for plotting the low frequency and data.
        :type plot_kwargs: dict
        :param hf_plot_kwargs: Arguments for the high frequency plotting.
        :type hf_plot_kwargs: dict
        :param plot_contours: Indicates if contours should be plotted.
        :type plot_contours: bool
        :return fig: figure handle for the plot
        :rtype fig: matplotlib.figure()
        :return axes: matplotlib subplot instances
        :rtype axes: matplotlib.Axes()
        """
        if scale_reconstruction_kwargs is None:
            scale_reconstruction_kwargs = {}

        xr_sep = self.scale_reconstruction(**scale_reconstruction_kwargs)

        if fig_kwargs is None:
            fig_kwargs = {}
        fig_kwargs["sharex"] = fig_kwargs.get("sharex", True)
        fig_kwargs["figsize"] = fig_kwargs.get(
            "figsize", (6, 1.5 * len(self._cluster_centroids) + 1)
        )

        # Low frequency and input data often require separate plotting
        # parameters.
        if plot_kwargs is None:
            plot_kwargs = {}
        plot_kwargs["vmin"] = plot_kwargs.get("vmin", -np.abs(data).max())
        plot_kwargs["vmax"] = plot_kwargs.get("vmax", np.abs(data).max())
        plot_kwargs["cmap"] = plot_kwargs.get("cmap", "cividis")

        # High frequency components often require separate plotting parameters.
        if hf_plot_kwargs is None:
            hf_plot_kwargs = {}
        hf_plot_kwargs["vmin"] = hf_plot_kwargs.get(
            "vmin", -np.abs(xr_sep[1:, :, :]).max()
        )
        hf_plot_kwargs["vmax"] = hf_plot_kwargs.get(
            "vmax", np.abs(xr_sep[1:, :, :]).max()
        )
        hf_plot_kwargs["cmap"] = hf_plot_kwargs.get("cmap", "RdBu_r")

        # Determine the number of plotting elements, which changes depending on
        # if the residual is included.
        if plot_residual:
            num_plot_elements = len(self._cluster_centroids) + 2
        else:
            num_plot_elements = len(self._cluster_centroids) + 1
        fig, axes = plt.subplots(
            num_plot_elements,
            1,
            **fig_kwargs,
        )

        ax = axes[0]
        ax.pcolormesh(data.real, **plot_kwargs)
        if plot_contours:
            ax.contour(data.real, colors=["k"])
        ax.set_ylabel("Space (-)")
        ax.set_xlabel("Time (-)")
        ax.set_title(
            f"Input Data at decomposition window length = {self._window_length}"
        )
        for n_cluster, cluster in enumerate(self._cluster_centroids):
            if plot_period:
                x = 2 * np.pi / cluster
                title = "Reconstruction, central period={:.2f}"
            else:
                x = cluster
                title = "Reconstruction, central eig={:.2f}"

            ax = axes[n_cluster + 1]
            xr_scale = xr_sep[n_cluster, :, :]
            if n_cluster == 0:
                ax.pcolormesh(xr_scale, **plot_kwargs)
                if plot_contours:
                    ax.contour(xr_scale, colors=["k"])
            else:
                ax.pcolormesh(xr_scale, **hf_plot_kwargs)
            ax.set_ylabel("Space (-)")
            ax.set_title(title.format(x))

        if plot_residual:
            ax = axes[-1]
            ax.set_title("Residual")
            ax.pcolormesh(data - xr_sep.sum(axis=0), **hf_plot_kwargs)
            ax.set_ylabel("Space (-)")

        axes[-1].set_xlabel("Time (-)")
        fig.tight_layout()

        return fig, axes

    def plot_error(
        self, data, scale_reconstruction_kwargs=None, plot_kwargs=None
    ):
        """Plots the error for the COSTS fit

        Plots are a space-time diagram assuming a 1D spatial dimension.

        Determining the error requires the input data and the fit will be
        reconstructed.

        :param data: Data on which COSTS was fit
        :type data: numpy.ndarray
        :param scale_reconstruction_kwargs: Arguments for reconstructing the
            fit.
        :type scale_reconstruction_kwargs: dict
        :param plot_kwargs: Arguments passed to costs.plot_error().
        :type scale_reconstruction_kwargs: dict
        :return fig: figure handle for the plot
        :rtype fig: matplotlib.figure()
        :return axes: matplotlib subplot instances
        :rtype axes: matplotlib.Axes()
        """
        if scale_reconstruction_kwargs is None:
            scale_reconstruction_kwargs = {}
        if plot_kwargs is None:
            plot_kwargs = {}
        plot_kwargs["vmin"] = plot_kwargs.get("vmin", -3)
        plot_kwargs["vmax"] = plot_kwargs.get("vmax", 3)
        plot_kwargs["cmap"] = plot_kwargs.get("cmap", "RdBu_r")

        global_reconstruction = self.global_reconstruction(
            scale_reconstruction_kwargs=scale_reconstruction_kwargs
        )

        fig_glbl_r, ax_glbl_r = plt.subplots(
            1,
            1,
        )
        im = ax_glbl_r.pcolormesh(
            (global_reconstruction.real - data.real) / data.real * 100,
            **plot_kwargs,
        )

        cbar = fig_glbl_r.colorbar(im)
        cbar.set_label("% error")

        ax_glbl_r.set_xlabel("time (-)")
        ax_glbl_r.set_ylabel("space (-)")
        re = self.relative_error(global_reconstruction.real, data)
        ax_glbl_r.set_title(f"Error in Global Reconstruction = {re:.2}")

    def plot_time_series(
        self,
        space_index,
        data,
        scale_reconstruction_kwargs=None,
        include_residual=False,
    ):
        """Plots CoSTS for a single spatial point.

        Includes the input data for decomposition, the low-frequency component
        for the next decomposition level, the residual of the high frequency
        component, and the reconstructions of the frequency bands for the point.

        :param space_index: Index of the point in space for the 1D snapshot.
        :type space_index: int
        :param data: Original data, only necessary for level=0.
        :type data: numpy.ndarray
        :param scale_reconstruction_kwargs: Arguments for reconstructing the
            fit.
        :type scale_reconstruction_kwargs: dict
        :param include_residual:
        :return fig: figure handle for the plot
        :rtype fig: matplotlib.figure()
        :return axes: matplotlib subplot instances
        :rtype axes: matplotlib.Axes()
        """
        ground_truth_mean = data.mean(axis=1)
        ground_truth = (data.T - ground_truth_mean).T
        ground_truth = ground_truth[space_index, :]
        ground_truth_mean = ground_truth_mean[space_index]

        if scale_reconstruction_kwargs is None:
            scale_reconstruction_kwargs = {}
        xr_sep = self.scale_reconstruction(**scale_reconstruction_kwargs)

        fig, axes = plt.subplots(
            nrows=self.n_components + 2,
            sharex=True,
            figsize=(8, np.max((8, 1.5 * self.n_components))),
        )
        # Only share the y axis for the components
        for target in axes[2:]:
            target._shared_axes["y"].join(target, axes[2])

        ax = axes[0]
        ax.plot(ground_truth, color="k")
        ax.plot(
            xr_sep.sum(axis=0)[space_index, :] - ground_truth_mean,
            color="r",
            lw=0.5,
        )
        ax.set_title(
            f"window={self._window_length}, black=input data, "
            f"red=reconstruction"
        )
        ax.set_ylabel("Amp.")
        ax.set_xlabel("Time")

        ax = axes[1]
        ax.plot(
            ground_truth - (xr_sep[1:, :, :].sum(axis=0))[space_index, :],
            color="k",
        )

        for n in range(self.n_components):
            ax = axes[n + 1]
            if n == 0:
                title = (
                    "blue = Low-frequency component, black = high "
                    "frequency residual"
                )
                ax.plot(xr_sep[n, space_index, :] - ground_truth_mean)
            else:
                period = 2 * np.pi / self.cluster_centroids[n]
                title = f"Band period = {period:.0f} window length"
                ax.plot(xr_sep[n, space_index, :])
            ax.set_title(title)
            ax.set_ylabel("Amp.")
            ax.set_xlabel("Time")

        ax = axes[-1]
        ax.plot(ground_truth, color="k", label="Smoothed data")
        ax.plot(
            (xr_sep[1:, :, :].sum(axis=0))[space_index, :],
            label="High-frequency",
        )
        ax.plot(
            xr_sep[0, space_index, :] - ground_truth_mean, label="Low-frequency"
        )
        if include_residual:
            ax.plot(
                ground_truth - xr_sep.sum(axis=0)[space_index, :],
                label="Residual",
            )
            ax.set_title(
                (
                    "black=input data, yellow=low-frequency, "
                    "blue=high-frequency, red=residual"
                )
            )
        else:
            ax.set_title(
                "black=input data, yellow=low-frequency, blue=high-frequency"
            )

        ax.set_ylabel("Amp.")
        ax.set_xlabel("Time")
        ax.set_xlim(0, self._n_time_steps)
        fig.tight_layout()

        return fig, axes

    def to_xarray(self):
        """Build an xarray dataset from the fitted CoSTS object.

        The CoSTS object is converted to an xarray dataset, which allows
        saving the computationally expensive results, e.g., between iterations.

        The reconstructed data are not included since their size can rapidly
        explode to unexpected sizes. e.g., a 30MB dataset, decomposed at 6
        levels with an average number of frequency bands across decomposition
        levels equal to 8 becomes 1.3GB once reconstructed for each band.

        The functions `to_xarray` and `from_xarray` should allow for a complete
        round trip of the COSTS object without alteration.

        :return: COSTS fit in xarray format
        :rtype: xarray.Dataset
        """
        ds = xr.Dataset(
            {
                "omega": (("window_time_means", "svd_rank"), self.omega_array),
                "omega_classes": (
                    ("window_time_means", "svd_rank"),
                    self.omega_classes,
                ),
                "amplitudes": (
                    ("window_time_means", "svd_rank"),
                    self.amplitudes_array,
                ),
                "modes": (
                    ("window_time_means", "space", "svd_rank"),
                    self.modes_array,
                ),
                "window_means": (
                    ("window_time_means", "space"),
                    self.window_means_array,
                ),
                "cluster_centroids": (
                    "frequency_band",
                    self._cluster_centroids,
                ),
            },
            coords={
                "window_time_means": np.mean(self.time_array, axis=1),
                "slide": ("window_time_means", np.arange(self._n_slides)),
                "svd_rank": np.arange(self.svd_rank),
                "space": np.arange(self._n_data_vars),
                "frequency_band": np.arange(self.n_components),
                "window_index": np.arange(self._window_length),
                "time": (
                    ("window_time_means", "window_index"),
                    self.time_array,
                ),
            },
            attrs={
                "svd_rank": self.svd_rank,
                "svd_rank_pre_allocate": self._svd_rank_pre_allocate,
                "omega_transformation": self._xarray_sanitize(
                    self._transform_method
                ),
                "n_slides": self._n_slides,
                "window_length": self._window_length,
                "num_frequency_bands": self.n_components,
                "n_data_vars": self._n_data_vars,
                "n_time_steps": self._n_time_steps,
                "step_size": self._step_size,
                "non_integer_n_slide": self._non_integer_n_slide,
                "global_svd": self._global_svd,
                "relative_filter_length": self._relative_filter_length,
                "kern_method": self._kern_method,
            },
        )

        for kw, kw_val in self._pydmd_kwargs.items():
            ds.attrs[f"pydmd_kwargs__{kw}"] = self._xarray_sanitize(kw_val)

        return ds

    def from_xarray(self, ds):
        """Convert xarray Dataset into a fitted CoSTS object.

        The functions `to_xarray` and `from_xarray` should allow for a complete
        round trip of the COSTS object without alteration.

        :return: Previously fitted COSTS object.
        """

        self._omega_array = ds.omega.values
        self._omega_classes = ds.omega_classes.values
        self._amplitudes_array = ds.amplitudes.values
        self._modes_array = ds.modes.values
        self._window_means_array = ds.window_means.values
        self._cluster_centroids = ds.cluster_centroids.values
        self._time_array = ds.time.values
        self._n_slides = ds.attrs["n_slides"]
        self._svd_rank = ds.attrs["svd_rank"]
        self._n_data_vars = ds.attrs["n_data_vars"]
        self._n_time_steps = ds.attrs["n_time_steps"]
        self._n_components = ds.attrs["num_frequency_bands"]
        self._non_integer_n_slide = ds.attrs["non_integer_n_slide"]
        self._step_size = ds.attrs["step_size"]
        self._window_length = ds.attrs["window_length"]
        self._global_svd = ds.attrs["global_svd"]
        self._relative_filter_length = ds.attrs["relative_filter_length"]
        self._kern_method = ds.attrs["kern_method"]
        self._svd_rank_pre_allocate = ds.attrs["svd_rank_pre_allocate"]

        self._pydmd_kwargs = {}
        for attr in ds.attrs:
            if "pydmd_kwargs" in attr:
                new_attr_name = attr.replace("pydmd_kwargs__", "")
                self._pydmd_kwargs[new_attr_name] = self._xarray_unsanitize(
                    ds.attrs[attr]
                )
                if new_attr_name == "eig_constraints":
                    self._pydmd_kwargs[new_attr_name] = set(
                        self._pydmd_kwargs[new_attr_name]
                    )
            elif "omega_transformation" in attr:
                self._transform_method = self._xarray_unsanitize(ds.attrs[attr])

        return self

    @staticmethod
    def _xarray_sanitize(value):
        """Handle Nones in the pydmd_kwargs (i.e., used default values)

        Netcdf cannot handle NoneTypes. To allow the xarray DataSet to be
        saved to file we have to "sanitize" the NoneTypes. These two functions
        allow for a round trip recovery of `pydmd_kwargs`.

        :param value: Value to be stored in the attributes of an xarray Dataset.
        :return: value unaltered except if value NoneType.
        """
        if value is None:
            value = "None"
        elif isinstance(value, set):
            value = list(value)
        elif callable(value):
            value = f"Custom function {value.__name__}"
        return value

    @staticmethod
    def _xarray_unsanitize(value):
        """Handle Nones in the pydmd_kwargs (i.e., used default values)

        Netcdf cannot handle NoneTypes. To allow the xarray DataSet to be
        saved to file we have to "sanitize" the NoneTypes. These two functions
        allow for a round trip recovery of `pydmd_kwargs`.

        :param value: Value stored in the attributes of an xarray Dataset.
        :return: value unaltered except if value is the string "None".
        """
        # To handle the varying behavior between python versions when evaluating
        # a mixed type statement we have to try to catch cases when `value` is
        # an array (e.g., `proj_basis` kwarg). The except block should not be
        # triggered but is meant to catch edge cases when the user provides
        # an unexpected type (e.g., tuple).
        if not hasattr(value, "shape"):
            try:
                if value == "None":
                    return None
            except ValueError:
                return value
        return value
