# pylint: disable=invalid-name,too-many-arguments,no-member
"""
Unit tests for functions that solve for delta-TEC variations across the array
"""
import numpy
import pytest
from astropy import constants as const
from astropy import units
from astropy.coordinates import SkyCoord
from ska_sdp_datamodels.calibration.calibration_create import (
    create_gaintable_from_visibility,
)
from ska_sdp_datamodels.configuration.config_create import (
    create_named_configuration,
)
from ska_sdp_datamodels.science_data_model import PolarisationFrame
from ska_sdp_datamodels.sky_model import SkyComponent
from ska_sdp_datamodels.visibility.vis_create import create_visibility

from ska_sdp_func_python.calibration.ionosphere_solvers import (
    apply_phase_distortions,
    build_normal_equation,
    get_param_count,
    set_cluster_maps,
    set_coeffs_and_params,
    solve_ionosphere,
    solve_normal_equation,
    update_gain_table,
)
from ska_sdp_func_python.calibration.ionosphere_utils import (
    decompose_phasescreen,
)
from ska_sdp_func_python.imaging.dft import dft_skycomponent_visibility

pytest.importorskip(
    modname="ska_sdp_func", reason="ska-sdp-func is an optional dependency"
)


@pytest.fixture(scope="module", name="input_params")
def input_for_ionosphere_solvers():
    """Fixture for the ionosphere_solvers unit tests"""
    # Generate an array comprising the core stations and a handful of
    # six-station clusters
    low_config = create_named_configuration("LOWBD2", rmax=1200.0)
    # reduce the number of core baselines in line with AA2
    low_config, cluster_id = down_select_core_stations(low_config)

    # single time step
    times = numpy.array([0])
    # A handful of channels across a few hundred MGz
    chanwidth = 25e6
    frequency = numpy.arange(100e6 + chanwidth / 2.0, 250e6, chanwidth)

    # Generate visibilities
    ra0 = 0.0
    dec0 = -27.0
    polarisation_frame = PolarisationFrame("stokesI")

    phase_centre = SkyCoord(
        ra=ra0 * units.hourangle,
        dec=dec0 * units.deg,
        frame="icrs",
        equinox="J2000",
    )

    modelvis = create_visibility(
        low_config,
        times,
        frequency,
        channel_bandwidth=[chanwidth] * len(frequency),
        polarisation_frame=polarisation_frame,
        phasecentre=phase_centre,
        weight=1.0,
    )

    # Generate a calibrator
    flux_density = 1.0 * (frequency / frequency[0]) ** (-0.8)

    cal_cmp = SkyComponent(
        direction=SkyCoord(
            ra=(ra0 + 0.01) * units.hourangle,
            dec=(dec0 + 0.5) * units.deg,
            frame="icrs",
            equinox="J2000",
        ),
        frequency=frequency,
        name="cal",
        flux=flux_density[:, numpy.newaxis],  # add the polarisation axis
        polarisation_frame=PolarisationFrame("stokesI"),
    )

    # Add calibrator to visibilities
    vis = dft_skycomponent_visibility(modelvis, cal_cmp)

    parameters = {
        "configuration": low_config,
        "visibility": vis,
        "cluster_id": cluster_id,
    }

    return parameters


def down_select_core_stations(low_config):
    """Helper function to reduce core stations in line with AA2"""
    n_stations = low_config.stations.shape[0]
    # some configs have more stations in the core, but LOWBD2 appears to have:
    n_core = 212
    # indices of stations in the AA2 core:
    core_stations = (
        numpy.array(
            [
                4,
                8,
                16,
                17,
                22,
                23,
                30,
                31,
                32,
                33,
                36,
                52,
                56,
                57,
                59,
                62,
                66,
                69,
                70,
                72,
                73,
                78,
                80,
                88,
                89,
                90,
                91,
                98,
                108,
                111,
                132,
                144,
                146,
                158,
                165,
                167,
                176,
                183,
                193,
                200,
            ]
        )
        - 1
    )
    mask = numpy.isin(low_config.id.data, core_stations) | (
        low_config.id.data >= n_core
    )

    low_config = low_config.sel(
        indexers={"id": numpy.arange(n_stations)[mask]}
    )

    # reset the station indices
    n_stations = low_config.stations.shape[0]
    low_config.stations.data = numpy.arange(n_stations).astype("str")
    low_config = low_config.assign_coords(id=numpy.arange(n_stations))

    n_core = len(core_stations)
    cluster_id = numpy.zeros(n_stations, "int")
    cluster_id[n_core:n_stations] = (
        numpy.arange(n_stations - n_core).astype("int") // 6 + 1
    )

    return low_config, cluster_id


def test_set_cluster_maps(input_params):
    """Unit tests for set_cluster_maps function:
    check that dimensions and values make sense
    """
    low_config = input_params["configuration"]
    cluster_id = input_params["cluster_id"]
    [n_cluster, cid2stn, stn2cid] = set_cluster_maps(cluster_id)
    # check vector to convert station indices to cluster IDs
    assert len(stn2cid) == len(low_config.stations)
    assert numpy.all(stn2cid >= 0)
    assert numpy.all(stn2cid < n_cluster)
    # check list to convert cluster IDs to vectors of station indices
    assert len(cid2stn) == n_cluster
    for cid in range(n_cluster):
        assert numpy.all(cid2stn[cid] >= 0)
        assert numpy.all(cid2stn[cid] < len(low_config.stations))


def test_set_coeffs_and_params(input_params):
    """Unit tests for set_coeffs_and_params function:
    check that dimensions and values make sense
    """
    low_config = input_params["configuration"]
    cluster_id = input_params["cluster_id"]
    [n_cluster, _, stn2cid] = set_cluster_maps(cluster_id)
    [param, coeff] = set_coeffs_and_params(low_config.xyz.data, cluster_id)
    # check that there is a set of parameters per cluster
    n_param_0 = 16
    n_param_n = 4
    assert len(param) == n_cluster
    assert len(param[0]) == n_param_0
    for cid in range(1, n_cluster):
        assert len(param[cid]) == n_param_n
    # check that there is a set of basis function values per stations
    assert len(coeff) == len(low_config.stations)
    for stn in range(len(low_config.stations)):
        if stn2cid[stn] == 0:
            assert len(param[0]) == n_param_0
        else:
            assert len(param[cid]) == n_param_n


def test_get_param_count(input_params):
    """Unit tests for get_param_count function:
    check that dimensions and values make sense
    """
    low_config = input_params["configuration"]
    cluster_id = input_params["cluster_id"]
    n_cluster = set_cluster_maps(cluster_id)[0]
    param = set_coeffs_and_params(low_config.xyz.data, cluster_id)[0]
    [n_param, pidx0] = get_param_count(param)
    # recalculate them here
    this_n_param = 0
    this_pidx0 = numpy.zeros(n_cluster, "int")
    for cid in range(n_cluster):
        this_pidx0[cid] = this_n_param
        this_n_param += len(param[cid])
    assert this_n_param == n_param
    assert numpy.all(this_pidx0 == pidx0)


def test_apply_phase_distortions(input_params):
    """Unit tests for apply_phase_distortions function:
    This is tested functionally in test_solve_ionosphere
    Here just check that dimensions and values make sense
    """
    low_config = input_params["configuration"]
    initialvis = input_params["visibility"]
    cluster_id = input_params["cluster_id"]
    [n_cluster, _, stn2cid] = set_cluster_maps(cluster_id)
    [param, coeff] = set_coeffs_and_params(low_config.xyz.data, cluster_id)
    # Set some random param values to apply. Not too big or the will be wraps
    numpy.random.seed(int(1e8))
    for cid in range(n_cluster):
        param[cid] = 0.01 * numpy.random.randn(len(param[cid]))

    finalvis = initialvis.copy(deep=True)
    apply_phase_distortions(finalvis, param, coeff, cluster_id)

    # shape should not have changed on output
    assert numpy.all(initialvis.vis.shape == finalvis.vis.shape)
    # amplitude should not have changed on output
    assert numpy.allclose(
        numpy.abs(initialvis.vis.data),
        numpy.abs(finalvis.vis.data),
        rtol=1e-12,
    )
    # some of the phases should have changed
    assert numpy.any(
        numpy.angle(initialvis.vis.data) != numpy.angle(finalvis.vis.data)
    )
    # tests one of the phase shifts
    idx = 26
    chan = 3
    stn1 = initialvis.antenna1.data[idx]
    stn2 = initialvis.antenna2.data[idx]
    assert (
        numpy.abs(
            numpy.angle(finalvis.vis.data[0, idx, chan, 0])
            - numpy.angle(initialvis.vis.data[0, idx, chan, 0])
            - 2.0
            * numpy.pi
            * const.c.value
            / initialvis.frequency.data[chan]
            * (
                numpy.sum(param[stn2cid[stn1]] * coeff[stn1])
                - numpy.sum(param[stn2cid[stn2]] * coeff[stn2])
            )
        )
        < 1e-12
    )


def test_build_normal_equation(input_params):
    """Unit tests for build_normal_equation function:
    This is tested functionally in test_solve_ionosphere
    Here just check that dimensions and values make sense
    """
    low_config = input_params["configuration"]
    modelvis = input_params["visibility"]
    cluster_id = input_params["cluster_id"]
    n_cluster = set_cluster_maps(cluster_id)[0]
    [param, coeff] = set_coeffs_and_params(low_config.xyz.data, cluster_id)
    n_param = get_param_count(param)[0]

    # Set some random param values to apply. Not too big or the will be wraps
    numpy.random.seed(int(1e8))
    param_true = []
    for cid in range(n_cluster):
        param_true.append(0.01 * numpy.random.randn(len(param[cid])))

    vis = modelvis.copy(deep=True)
    apply_phase_distortions(vis, param_true, coeff, cluster_id)

    [AA, Ab] = build_normal_equation(vis, modelvis, param, coeff, cluster_id)
    assert AA.shape[0] == n_param
    assert AA.shape[1] == n_param
    assert Ab.shape[0] == n_param


def test_solve_normal_equation(input_params):
    """Unit tests for solve_normal_equation function:
    This is tested functionally in test_solve_ionosphere
    Here just check that dimensions and values make sense
    """
    low_config = input_params["configuration"]
    modelvis = input_params["visibility"]
    cluster_id = input_params["cluster_id"]
    [n_cluster, _, stn2cid] = set_cluster_maps(cluster_id)
    [param, coeff] = set_coeffs_and_params(low_config.xyz.data, cluster_id)

    # Set some random param values to apply. Not too big or the will be wraps
    numpy.random.seed(int(1e8))
    param_true = []
    for cid in range(n_cluster):
        param_true.append(0.0001 * numpy.random.randn(len(param[cid])))

    vis = modelvis.copy(deep=True)
    apply_phase_distortions(vis, param_true, coeff, cluster_id)

    [AA, Ab] = build_normal_equation(vis, modelvis, param, coeff, cluster_id)

    # Solve the normal equations and update parameters
    solve_normal_equation(AA, Ab, param, 0)

    # tests one of the phase shifts
    # idx = 26
    # chan = 3
    stn1 = modelvis.antenna1.data[26]
    stn2 = modelvis.antenna2.data[26]
    # the factor nu should match the value in solve_normal_equation
    # nu = 0.5
    assert (
        numpy.abs(
            0.5
            * (
                numpy.angle(vis.vis.data[0, 26, 3, 0])
                - numpy.angle(modelvis.vis.data[0, 26, 3, 0])
            )
            - 2.0
            * numpy.pi
            * const.c.value
            / modelvis.frequency.data[3]
            * (
                numpy.sum(param[stn2cid[stn1]] * coeff[stn1])
                - numpy.sum(param[stn2cid[stn2]] * coeff[stn2])
            )
        )
        < 1e-5
    )


def test_update_gain_table(input_params):
    """Unit tests for update_gain_table function:
    This is tested functionally in test_solve_ionosphere
    Here just check that dimensions and values make sense
    """
    low_config = input_params["configuration"]
    vis = input_params["visibility"]
    cluster_id = input_params["cluster_id"]
    [n_cluster, _, stn2cid] = set_cluster_maps(cluster_id)
    [param, coeff] = set_coeffs_and_params(low_config.xyz.data, cluster_id)
    # Set some random param values to apply. Not too big or the will be wraps
    numpy.random.seed(int(1e8))
    for cid in range(n_cluster):
        param[cid] = 0.01 * numpy.random.randn(len(param[cid]))

    initial_table = create_gaintable_from_visibility(vis, jones_type="B")
    final_table = initial_table.copy(deep=True)
    update_gain_table(final_table, param, coeff, cluster_id)

    # shape should not have changed on output
    assert numpy.all(initial_table.gain.shape == final_table.gain.shape)
    # amplitude should not have changed on output
    assert numpy.allclose(
        numpy.abs(initial_table.gain.data),
        numpy.abs(final_table.gain.data),
        rtol=1e-12,
    )
    # some of the phases should have changed
    assert numpy.any(
        numpy.angle(initial_table.gain.data)
        != numpy.angle(final_table.gain.data)
    )
    # tests one of the phase shifts
    idx = 26
    chan = 3
    stn = initial_table.antenna.data[idx]
    assert (
        numpy.abs(
            numpy.angle(final_table.gain.data[0, idx, chan, 0, 0])
            - numpy.angle(initial_table.gain.data[0, idx, chan, 0, 0])
            - 2.0
            * numpy.pi
            * const.c.value
            / initial_table.frequency.data[chan]
            * numpy.sum(param[stn2cid[stn]] * coeff[stn])
        )
        < 1e-12
    )


def test_solve_ionosphere(input_params):
    """Unit tests for solve_ionosphere function:
    check that grid_data is updated
    """
    low_config = input_params["configuration"]
    modelvis = input_params["visibility"]
    cluster_id = input_params["cluster_id"]

    # Make a "true" copy of the visibilties and apply ionospheric distortions
    vis = modelvis.copy(deep=True)

    # Generate station-based phase shifts
    freq_0 = 150e6
    screen = generate_phase_shifts(low_config)

    # Generate the matrix of visibilty phase shifts at freq_0 then select
    # just the relevant baselines (cross and auto correlations)
    n_stations = low_config.stations.shape[0]
    stn = numpy.arange(n_stations)
    mask = numpy.tile(stn[:, numpy.newaxis], (1, n_stations)) <= numpy.tile(
        stn[numpy.newaxis, :], (n_stations, 1)
    )
    vis_phase = (
        numpy.tile(screen[:, numpy.newaxis], (1, n_stations))
        - numpy.tile(screen[numpy.newaxis, :], (n_stations, 1))
    )[mask]
    # Apply ionospheric phase shifts to the true visibilities
    wl_scaling = freq_0 / vis.frequency.data
    assert vis.vis.data.shape[1] == len(
        vis_phase
    ), f"second dimension of {vis.vis.data.shape} != {len(vis_phase)}"
    assert vis.vis.data.shape[2] == len(
        wl_scaling
    ), f"third dimension of {vis_phase.shape} != {len(wl_scaling)}"
    vis.vis.data[0, :, :, 0] *= numpy.exp(
        1j * numpy.einsum("b,f->bf", vis_phase, wl_scaling)
    )

    updatedvis = modelvis.copy(deep=True)

    gain_table = solve_ionosphere(
        vis,
        updatedvis,
        low_config.xyz.data,
        cluster_id,
    )
    assert len(gain_table.frequency) == len(vis.frequency)
    assert numpy.all(numpy.abs(numpy.abs(gain_table.gain.data) - 1.0) < 1e-12)

    # Get the fitted phase shifts, scaled to the screen frequency
    fit = numpy.angle(gain_table.gain.data[0, :, 0, 0, 0]) / wl_scaling[0]

    # Phase referencing
    # against a single station:
    # screen -= screen[0]
    # fit -= fit[0]
    # or against all of the core stations:
    screen -= numpy.mean(screen[cluster_id == 0])
    fit -= numpy.mean(fit[cluster_id == 0])

    # r2d = 180.0 / numpy.pi
    # print(f"max error = {numpy.amax(numpy.abs(screen - fit))} rad")
    # print(f"error RMS = {numpy.sqrt(numpy.mean((screen - fit)**2))} rad")

    assert numpy.amax(numpy.abs(screen - fit)) < 0.02277
    assert numpy.sqrt(numpy.mean((screen - fit) ** 2)) < 0.00621


def generate_phase_shifts(low_config, r_0=7e3, beta=5.0 / 3.0):
    """Helper function to generate station-based phase shifts

    Generate turbulent phase shifts for each station with a diffractive scale
    of r_0 metres at freq0 Hz and a spatial power law exponent of beta
    """
    n_stations = low_config.stations.shape[0]
    x_stn = low_config.xyz.data[:, 0]
    y_stn = low_config.xyz.data[:, 1]

    [evec_matrix, sqrt_evals] = decompose_phasescreen(x_stn, y_stn, r_0, beta)

    numpy.random.seed(int(1e8))
    return evec_matrix @ (sqrt_evals * numpy.random.randn(n_stations))
