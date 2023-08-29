# pylint: disable=invalid-name
"""
Unit tests for ionospheric calibration and phase screen utils
"""
import numpy
import pytest
from ska_sdp_datamodels.configuration.config_create import (
    create_named_configuration,
)

from ska_sdp_func_python.calibration.ionosphere_utils import (
    decompose_phasescreen,
    displace_phasescreen,
    interpolate_phasescreen,
    standardise_eigenvectors,
    zern,
    zern_array,
)


@pytest.fixture(scope="module", name="input_params")
def input_for_ionosphere_util():
    """Fixture for the ionosphere_util unit tests"""
    # Generate an array comprising the core stations and a handful of
    # six-station clusters
    low_config = create_named_configuration("LOWBD2", rmax=1200.0)

    parameters = {
        "configuration": low_config,
    }

    return parameters


def test_decompose_phasescreen(input_params):
    """Unit tests for decompose_phasescreen function:
    Generate many random screens and fit the structure function parameters
    to the phase variance across the baselines. Make sure they are within
    a few percent of the input structure function parameters
    """
    # pylint: disable=too-many-locals
    low_config = input_params["configuration"]

    # Generate ionospheric phase shifts with Kolmogorov turbulence for each
    # station towards an unresolved radio source
    x = low_config.xyz.data[:, 0]
    y = low_config.xyz.data[:, 1]

    # Decompose the pierce point convariance matrix
    r_0 = 7e3
    beta = 5.0 / 3.0
    [evec_matrix, sqrt_evals] = decompose_phasescreen(x, y, r_0, beta)
    standardise_eigenvectors(evec_matrix)

    # Generate a mask to select cross products (c.f. baselines) from
    # n_stations x n_stations covariance matrices and the like
    n_stations = low_config.stations.shape[0]
    assert len(sqrt_evals) == n_stations
    idx = numpy.arange(n_stations)
    mask = numpy.tile(idx[:, numpy.newaxis], (1, n_stations)) > numpy.tile(
        idx[numpy.newaxis, :], (n_stations, 1)
    )

    # Calculate the separation of station pairs. Pierce points towards a point
    # on the celestial sphere through a 2D horizontal ionosphere will have the
    # same relative positions
    u = (
        numpy.tile(x[:, numpy.newaxis], (1, n_stations))
        - numpy.tile(x[numpy.newaxis, :], (n_stations, 1))
    )[mask]
    v = (
        numpy.tile(y[:, numpy.newaxis], (1, n_stations))
        - numpy.tile(y[numpy.newaxis, :], (n_stations, 1))
    )[mask]
    r = numpy.sqrt(u * u + v * v)

    # Generate the random screens and accumulate the covariance estimates
    shift_variance = numpy.zeros(len(r))
    numpy.random.seed(int(1e8))
    num_screens = 2000
    for _ in range(num_screens):
        phase_shifts = evec_matrix @ (
            sqrt_evals * numpy.random.randn(n_stations)
        )
        relative_shifts = (
            numpy.tile(phase_shifts[:, numpy.newaxis], (1, n_stations))
            - numpy.tile(phase_shifts[numpy.newaxis, :], (n_stations, 1))
        )[mask]
        shift_variance += relative_shifts**2 / num_screens

    # Fit structure function parameters from the covariance measurements
    r_0_fit, beta_fit = fit_structure_function(r, shift_variance)

    assert numpy.abs(beta_fit - beta) / beta < 5e-3
    assert numpy.abs(r_0_fit - r_0) / r_0 < 2e-2


def test_decompose_phasescreen_array():
    """Unit tests for decompose_phasescreen function:
    Like test_decompose_phasescreen but with 2D phase screen input
    """
    # pylint: disable=too-many-locals
    # Generate a Nside x Nside grid of ionospheric phase shifts with
    # Kolmogorov turbulence
    res = 500.0
    Nside = 21
    x = numpy.arange(Nside) * res
    y = numpy.arange(Nside) * res
    xx, yy = numpy.meshgrid(x, y)

    # Decompose the pierce point convariance matrix
    r_0 = 7e3
    beta = 5.0 / 3.0
    [evec_matrix, sqrt_evals] = decompose_phasescreen(xx, yy, r_0, beta)
    standardise_eigenvectors(evec_matrix)

    # Generate a mask to select cross products (c.f. baselines) from
    # n_points x n_points covariance matrices and the like
    n_points = xx.size
    assert len(sqrt_evals) == n_points
    idx = numpy.arange(n_points)
    mask = numpy.tile(idx[:, numpy.newaxis], (1, n_points)) > numpy.tile(
        idx[numpy.newaxis, :], (n_points, 1)
    )

    # Calculate the separation of pierce-point pairs
    u = (
        numpy.tile(xx.reshape(n_points)[:, numpy.newaxis], (1, n_points))
        - numpy.tile(xx.reshape(n_points)[numpy.newaxis, :], (n_points, 1))
    )[mask]
    v = (
        numpy.tile(yy.reshape(n_points)[:, numpy.newaxis], (1, n_points))
        - numpy.tile(yy.reshape(n_points)[numpy.newaxis, :], (n_points, 1))
    )[mask]
    r = numpy.sqrt(u * u + v * v)

    # Generate the random screens and accumulate the covariance estimates
    shift_variance = numpy.zeros(len(r))
    numpy.random.seed(int(1e8))
    num_screens = 2000
    for _ in range(num_screens):
        phasescreen = evec_matrix @ (sqrt_evals * numpy.random.randn(n_points))
        relative_shifts = (
            numpy.tile(phasescreen[:, numpy.newaxis], (1, n_points))
            - numpy.tile(phasescreen[numpy.newaxis, :], (n_points, 1))
        )[mask]
        shift_variance += relative_shifts**2 / num_screens

    # Fit structure function parameters from the covariance measurements
    r_0_fit, beta_fit = fit_structure_function(r, shift_variance)

    assert numpy.abs(beta_fit - beta) / beta < 1e-2
    assert numpy.abs(r_0_fit - r_0) / r_0 < 2e-2


def test_interpolate_phasescreen():
    """Unit tests for interpolate_phasescreen function:
    Check that dimensions make sense
    """
    # Generate a Nside0 x Nside0 grid
    res0 = 500.0
    Nside0 = 21
    x = res0 * numpy.arange(Nside0)
    y = res0 * numpy.arange(Nside0)
    xx = numpy.meshgrid(x, y)[0]

    screen = numpy.zeros(xx.shape)

    Nside = Nside0
    res = res0

    screen = interpolate_phasescreen(screen)
    Nside = 2 * Nside - 1
    res /= 2.0

    assert screen.shape[0] == Nside
    assert numpy.all(screen == 0)


def test_displace_phasescreen():
    """Unit tests for displace_phasescreen function:
    Generate many random screens and fit structure function parameters to the
    phase covariance of random pairs of pierce points. Choose pierce point
    pairs with small separations so that the measured structure function is
    dominated by interpolated points with midpoint displacement. Make sure the
    fitted structure function parameters are within a few percent of the input
    values.
    """
    # pylint: disable=too-many-locals
    # Generate a Nside0 x Nside0 grid of ionospheric phase shifts with
    # Kolmogorov turbulence. decompose_phasescreen scales poorly with the
    # size of x and y, so generate a coarse grid and then interpolate it
    # using interpolate_phasescreen and displace_phasescreen.
    res0 = 500.0
    Nside0 = 21
    x = res0 * numpy.arange(Nside0)
    y = res0 * numpy.arange(Nside0)
    xx, yy = numpy.meshgrid(x, y)

    # Decompose the pierce point convariance matrix
    r_0 = 7e3
    beta = 5.0 / 3.0
    [evec_matrix, sqrt_evals] = decompose_phasescreen(xx, yy, r_0, beta)
    n_points = xx.size
    assert len(sqrt_evals) == n_points

    # Will successively interpolate the screen n_interp times
    n_interp = 4
    Nside = Nside0
    for _ in range(0, n_interp):
        Nside = 2 * Nside - 1

    # Choose pairs of screen pixels at which to calculate phase variance.
    # Choose pairs with small separations, i.e. with phase variations
    # dominated by interpolation and midpoint displacement.
    n_selection = 200
    max_pixel_sep = 2**n_interp

    # choose the first pierce points, with a buffer around the edge
    numpy.random.seed(int(1e8))
    i1 = max_pixel_sep + (
        (Nside - 2 * max_pixel_sep) * numpy.random.rand(n_selection)
    ).astype("int")
    j1 = max_pixel_sep + (
        (Nside - 2 * max_pixel_sep) * numpy.random.rand(n_selection)
    ).astype("int")
    # choose the second pierce points
    i2 = i1 + (
        max_pixel_sep * (2 * numpy.random.rand(n_selection) - 1)
    ).astype("int")
    j2 = j1 + (
        max_pixel_sep * (2 * numpy.random.rand(n_selection) - 1)
    ).astype("int")

    # Generate the random screens and accumulate the covariance estimates
    covariance = numpy.zeros(n_selection)
    for _ in range(200):
        # Generate the coarse starting screen
        screen = evec_matrix @ (sqrt_evals * numpy.random.randn(n_points))
        screen = screen.reshape(xx.shape)
        Nside = Nside0
        res = res0
        # Successively interpolate the screen and randomise the interpolated
        # points using midpoint displacement
        for _ in range(0, n_interp):
            screen = interpolate_phasescreen(screen)
            screen = displace_phasescreen(screen, res, r_0, beta)
            Nside = 2 * Nside - 1
            res /= 2.0
        covariance += (screen[i1, j1] - screen[i2, j2]) ** 2 / 200.0

    # Calculate the pierce-point pair separations
    x = res * numpy.arange(-(Nside // 2), Nside // 2 + 1)
    y = res * numpy.arange(-(Nside // 2), Nside // 2 + 1)
    r = numpy.sqrt((x[i1] - x[i2]) ** 2 + (y[j1] - y[j2]) ** 2)

    # Fit structure function parameters from the covariance measurements
    mask = covariance > 0
    r_0_fit, beta_fit = fit_structure_function(r[mask], covariance[mask])

    assert numpy.abs(beta_fit - beta) / beta < 5e-2
    assert numpy.abs(r_0_fit - r_0) / r_0 < 10e-2


def fit_structure_function(separation, covariance):
    """Helper function to fit structure function parameters

    Estimate structure function parameters from the covariance measurements
    covariance(separation) = (separation/r_0)**beta
     => log(covariance(separation)) = beta*log(separation) - beta*log(r_0)

    """
    assert len(separation) == len(covariance)
    beta, y0 = numpy.polyfit(numpy.log(separation), numpy.log(covariance), 1)
    return numpy.exp(-y0 / beta), beta


def test_zern():
    """Unit tests for zern function:
    Very basic checks
    """

    # pick n points uniformly distributed across a unit circle
    numpy.random.seed(int(1e8))
    n = 1000
    rho = numpy.sqrt(numpy.random.rand(n))
    phi = 2.0 * numpy.pi * numpy.random.rand(n)

    count = 0
    for n in range(5):
        for m in range(-n, n + 1, 2):
            coeff = zern(m, n, rho, phi)
            assert numpy.amin(coeff) >= -1
            assert numpy.amax(coeff) <= 1
            count += 1


def test_zern_array():
    """Unit tests for zern_array function:
    Very basic checks
    """

    # pick n points uniformly distributed across a square
    numpy.random.seed(int(1e8))
    n = 1000
    x = 30 * numpy.random.rand(n)
    y = 30 * numpy.random.rand(n)

    assert len(zern_array(0, x, y, noll_order=True)) == len(x)
    assert len(zern_array(0, x, y, noll_order=True)[0]) == 1
    assert len(zern_array(1, x, y, noll_order=True)[0]) == 3
    assert len(zern_array(2, x, y, noll_order=True)[0]) == 6
    assert len(zern_array(3, x, y, noll_order=True)[0]) == 10
    assert len(zern_array(4, x, y, noll_order=True)[0]) == 15

    assert len(zern_array(0, x, y, noll_order=False)) == len(x)
    assert len(zern_array(0, x, y, noll_order=False)[0]) == 1
    assert len(zern_array(2, x, y, noll_order=False)[0]) == 4
    assert len(zern_array(4, x, y, noll_order=False)[0]) == 9
    assert len(zern_array(6, x, y, noll_order=False)[0]) == 16
