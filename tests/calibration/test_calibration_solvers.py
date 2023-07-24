"""
Unit tests for calibration solution
"""
import numpy
import pytest
from ska_sdp_datamodels.calibration.calibration_create import (
    create_gaintable_from_visibility,
)

pytest.importorskip(
    modname="ska_sdp_func", reason="ska-sdp-func is an optional dependency"
)
from ska_sdp_func_python.calibration.operations import apply_gaintable
from ska_sdp_func_python.calibration.solvers import (
    find_best_refant_from_vis,
    solve_gaintable,
)
from tests.testing_utils import simulate_gaintable, vis_with_component_data


@pytest.mark.parametrize(
    "sky_pol_frame, data_pol_frame, flux_array, "
    "phase_error, expected_gain_sum",
    [
        (
            "stokesIQUV",
            "circular",
            [100.0, 0.0, 0.0, 50.0],
            10.0,
            (-2.3575149649, -19.50250306305245),
        ),
        (
            "stokesIQUV",
            "linear",
            [100.0, 50.0, 0.0, 0.0],
            10.0,
            (-2.3575149649, -19.50250306305245),
        ),
        (
            "stokesIV",
            "circularnp",
            [100.0, 50.0],
            0.1,
            (745.2361964760858, 23.999242596777464),
        ),
        (
            "stokesIQ",
            "linearnp",
            [100.0, 50.0],
            0.1,
            (745.2361964760858, 23.999242596777464),
        ),
        (
            "stokesIQUV",
            "circular",
            [100.0, 0.0, 0.0, 50.0],
            0.1,
            (745.2361964760858, 23.999242596777464),
        ),
        (
            "stokesIQUV",
            "linear",
            [100.0, 50.0, 0.0, 0.0],
            0.1,
            (745.2361964760858, 23.999242596777464),
        ),
        (
            "stokesI",
            "stokesI",
            [100.0, 0.0, 0.0, 0.0],
            0.1,
            (372.3578195252, 23.5790571427),
        ),
    ],
)
def test_solve_gaintable_phase_only(
    sky_pol_frame, data_pol_frame, flux_array, phase_error, expected_gain_sum
):
    """
    Test solve_gaintable for phase solution only (with phase_errors),
    for different polarisation frames.
    """
    jones_type = "T"

    vis = vis_with_component_data(sky_pol_frame, data_pol_frame, flux_array)

    gain_table = create_gaintable_from_visibility(vis, jones_type=jones_type)
    gain_table = simulate_gaintable(
        gain_table,
        phase_error=phase_error,
        amplitude_error=0.0,
        leakage=0.0,
    )

    original = vis.copy(deep=True)
    vis = apply_gaintable(vis, gain_table)

    result_gain_table = solve_gaintable(
        vis,
        original,
        phase_only=True,
        niter=200,
        crosspol=False,
        tol=1e-6,
        normalise_gains=None,
        jones_type=jones_type,
    )

    assert result_gain_table["gain"].data.sum().real.round(10) == round(
        expected_gain_sum[0], 10
    )
    assert result_gain_table["gain"].data.sum().imag.round(10) == -round(
        expected_gain_sum[1], 10
    )


@pytest.mark.parametrize(
    "sky_pol_frame, data_pol_frame, flux_array, "
    "amplitude_error, expected_gain_sum",
    [
        (
            "stokesIV",
            "circularnp",
            [100.0, 50.0],
            0.01,
            (745.4165220744279, -23.978608965851272),
        ),
        (
            "stokesIQUV",
            "circular",
            [100.0, 0.0, 0.0, 50.0],
            0.01,
            (745.4165220744279, -23.978608965851272),
        ),
        (
            "stokesIQUV",
            "linear",
            [100.0, 50.0, 0.0, 0.0],
            0.01,
            (745.4165220744279, -23.978608965851272),
        ),
        (
            "stokesI",
            "stokesI",
            [100.0, 0.0, 0.0, 0.0],
            0.1,
            (372.30912577554244, -23.891788423254088),
        ),
    ],
)
def test_solve_gaintable_phase_and_amplitude(
    sky_pol_frame,
    data_pol_frame,
    flux_array,
    amplitude_error,
    expected_gain_sum,
):
    """
    Test solve_gaintable with with phase and amplitude errors,
    for different polarisation frames.
    """
    jones_type = "G"

    vis = vis_with_component_data(sky_pol_frame, data_pol_frame, flux_array)

    gt = create_gaintable_from_visibility(vis, jones_type=jones_type)
    gt = simulate_gaintable(
        gt,
        phase_error=0.1,
        amplitude_error=amplitude_error,
        leakage=0.0,
    )
    original = vis.copy(deep=True)
    vis = apply_gaintable(vis, gt)

    result_gain_table = solve_gaintable(
        vis,
        original,
        phase_only=False,
        niter=200,
        crosspol=False,
        tol=1e-6,
        normalise_gains=None,
        jones_type=jones_type,
    )

    assert result_gain_table["gain"].data.sum().real.round(10) == round(
        expected_gain_sum[0], 10
    )
    assert result_gain_table["gain"].data.sum().imag.round(10) == round(
        expected_gain_sum[1], 10
    )


@pytest.mark.parametrize(
    "sky_pol_frame, data_pol_frame, flux_array",
    [
        ("stokesIQ", "linearnp", [100.0, 50.0]),
        ("stokesIQUV", "circular", [100.0, 10.0, -20.0, 50.0]),
        ("stokesIQUV", "circular", [100.0, 0.0, 0.0, 50.0]),
        ("stokesIQUV", "linear", [100.0, 50.0, 10.0, -20.0]),
        ("stokesIQUV", "linear", [100.0, 50.0, 0.0, 0.0]),
    ],
)
def test_solve_gaintable_crosspol(sky_pol_frame, data_pol_frame, flux_array):
    """
    Test solve_gaintable with crosspol=True, for different polarisation frames.
    """
    jones_type = "G"

    vis = vis_with_component_data(sky_pol_frame, data_pol_frame, flux_array)

    gt = create_gaintable_from_visibility(vis, jones_type=jones_type)
    gt = simulate_gaintable(
        gt,
        phase_error=0.1,
        amplitude_error=0.01,
        leakage=0.0,
    )
    original = vis.copy(deep=True)
    vis = apply_gaintable(vis, gt)

    result_gain_table = solve_gaintable(
        vis,
        original,
        phase_only=False,
        niter=200,
        crosspol=True,
        tol=1e-6,
        normalise_gains=None,
        jones_type=jones_type,
    )

    assert result_gain_table["gain"].data.sum().real.round(10) == round(
        745.4165220744279, 10
    )
    assert result_gain_table["gain"].data.sum().imag.round(10) == round(
        -23.978608965851272, 10
    )


def test_solve_gaintable_timeslice():
    """
    Test solve_gaintable with timeslice set.
    """
    jones_type = "G"

    vis = vis_with_component_data("stokesI", "stokesI", [100.0, 0.0, 0.0, 0.0])

    gt = create_gaintable_from_visibility(vis, jones_type=jones_type)
    gt = simulate_gaintable(
        gt,
        phase_error=0.1,
        amplitude_error=0.1,
        leakage=0.0,
    )
    original = vis.copy(deep=True)
    vis = apply_gaintable(vis, gt)

    result_gain_table = solve_gaintable(
        vis,
        original,
        phase_only=False,
        niter=200,
        crosspol=False,
        tol=1e-6,
        normalise_gains=None,
        jones_type=jones_type,
        timeslice=120.0,
    )

    assert result_gain_table["gain"].data.sum().real == 94.0
    assert result_gain_table["gain"].data.sum().imag == 0.0j


def test_solve_gaintable_normalise():
    """
    Test solve_gaintable with normalise_gains="mean".
    """
    jones_type = "G"

    vis = vis_with_component_data("stokesI", "stokesI", [100.0, 0.0, 0.0, 0.0])

    gt = create_gaintable_from_visibility(vis, jones_type=jones_type)
    gt = simulate_gaintable(
        gt,
        phase_error=0.1,
        amplitude_error=0.1,
        leakage=0.0,
    )
    original = vis.copy(deep=True)
    vis = apply_gaintable(vis, gt)

    result_gain_table = solve_gaintable(
        vis,
        original,
        phase_only=False,
        niter=200,
        crosspol=False,
        tol=1e-6,
        normalise_gains="mean",
        jones_type=jones_type,
    )

    assert (
        result_gain_table["gain"].data.sum().real.round(10) == 372.3183599042
    )
    assert (
        result_gain_table["gain"].data.sum().imag.round(10) == -23.8923809949
    )


@pytest.mark.parametrize(
    "sky_pol_frame, data_pol_frame, flux_array, "
    "crosspol, nchan, expected_gain_sum",
    [
        (
            "stokesI",
            "stokesI",
            [100.0, 0.0, 0.0, 0.0],
            False,
            32,
            (11920.084395699654, 2.887044851959022),
        ),
        (
            "stokesIQUV",
            "circular",
            [100.0, 0.0, 0.0, 50.0],
            True,
            4,
            (5961.616892881827, -4.6464727494239),
        ),
        (
            "stokesIQUV",
            "linear",
            [100.0, 50.0, 0.0, 0.0],
            False,
            32,
            (47695.42885072535, -290.36562000082),
        ),
    ],
)
def test_solve_gaintable_bandpass(
    sky_pol_frame,
    data_pol_frame,
    flux_array,
    crosspol,
    nchan,
    expected_gain_sum,
):
    """
    Test solve_gaintable for bandpass solution of multiple channels,
    for different polarisation frames.
    """
    jones_type = "B"

    vis = vis_with_component_data(
        sky_pol_frame, data_pol_frame, flux_array, nchan=nchan
    )

    gt = create_gaintable_from_visibility(vis, jones_type=jones_type)
    gt = simulate_gaintable(
        gt,
        phase_error=0.1,
        amplitude_error=0.1,
        leakage=0.0,
    )
    original = vis.copy(deep=True)
    vis = apply_gaintable(vis, gt)

    result_gain_table = solve_gaintable(
        vis,
        original,
        phase_only=False,
        niter=200,
        crosspol=crosspol,
        tol=1e-6,
        normalise_gains="mean",
        jones_type=jones_type,
    )

    assert result_gain_table["gain"].data.sum().real.round(10) == round(
        expected_gain_sum[0], 10
    )
    assert result_gain_table["gain"].data.sum().imag.round(10) == -round(
        expected_gain_sum[1], 10
    )


@pytest.mark.parametrize(
    "sky_pol_frame, data_pol_frame, flux_array, "
    "crosspol, nchan, expected_gain_sum",
    [
        (
            "stokesI",
            "stokesI",
            [100.0, 0.0, 0.0, 0.0],
            False,
            32,
            (11982.513069270433, 2.9021650788520645),
        ),
        (
            "stokesIQUV",
            "circular",
            [100.0, 0.0, 0.0, 50.0],
            True,
            4,
            (8559.667870101628, -6.671388688198775),
        ),
        (
            "stokesIQUV",
            "linear",
            [100.0, 50.0, 0.0, 0.0],
            False,
            32,
            (74274.34517898227, -452.1757494947468),
        ),
    ],
)
def test_solve_gaintable_bandpass_with_median(
    sky_pol_frame,
    data_pol_frame,
    flux_array,
    crosspol,
    nchan,
    expected_gain_sum,
):
    """
    Test solve_gaintable for bandpass solution of multiple channels,
    for different polarisation frames.
    """
    jones_type = "B"

    vis = vis_with_component_data(
        sky_pol_frame, data_pol_frame, flux_array, nchan=nchan
    )

    gt = create_gaintable_from_visibility(vis, jones_type=jones_type)
    gt = simulate_gaintable(
        gt,
        phase_error=0.1,
        amplitude_error=0.1,
        leakage=0.0,
    )
    original = vis.copy(deep=True)
    vis = apply_gaintable(vis, gt)

    result_gain_table = solve_gaintable(
        vis,
        original,
        phase_only=False,
        niter=200,
        crosspol=crosspol,
        tol=1e-6,
        normalise_gains="median",
        jones_type=jones_type,
    )

    assert result_gain_table["gain"].data.sum().real.round(10) == round(
        expected_gain_sum[0], 10
    )
    assert result_gain_table["gain"].data.sum().imag.round(10) == -round(
        expected_gain_sum[1], 10
    )


def test_solve_gaintable_few_antennas_many_times():
    """
    Test solve_gaintable for different array size and time samples.
    (Small array, large number of time samples)
    """
    jones_type = "G"

    vis = vis_with_component_data(
        "stokesI", "stokesI", [100.0, 0.0, 0.0, 0.0], rmax=83, ntimes=400
    )

    gt = create_gaintable_from_visibility(vis, jones_type=jones_type)
    gt = simulate_gaintable(
        gt,
        phase_error=0.1,
        amplitude_error=0.1,
        leakage=0.0,
    )
    original = vis.copy(deep=True)
    vis = apply_gaintable(vis, gt)

    result_gain_table = solve_gaintable(
        vis,
        original,
        phase_only=False,
        niter=200,
        crosspol=False,
        tol=1e-6,
        normalise_gains=None,
        jones_type=jones_type,
    )

    assert (
        result_gain_table["gain"].data.sum().real.round(10) == 2393.9044547675
    )
    assert (
        result_gain_table["gain"].data.sum().imag.round(10) == -24.2584035058
    )


def test_find_best_refant_from_vis():
    """
    Test best reference antenna under multiple frequencies
    """
    vis = vis_with_component_data(
        "stokesI", "stokesI", [100.0, 0.0, 0.0, 0.0], nchan=10
    )

    refant_sort = find_best_refant_from_vis(vis)
    # According to the input visibilities, we calculate the peak to
    # noise ratio (PNR) for FFT results in channel axis. We then
    # sort the PNR (decreased sort) to get a reference antenna
    # candidates list.  The first antenna would have the
    # maximum PNR.
    assert (refant_sort[:5] == numpy.array([53, 13, 49, 17, 56])).all()


def test_find_best_refant_from_vis_single_channel():
    """
    Test best reference antenna under single frequency
    """
    vis = vis_with_component_data(
        "stokesI", "stokesI", [100.0, 0.0, 0.0, 0.0], nchan=1
    )
    refant_sort = find_best_refant_from_vis(vis)
    assert (refant_sort[:5] == numpy.array([0, 1, 2, 3, 4])).all()
