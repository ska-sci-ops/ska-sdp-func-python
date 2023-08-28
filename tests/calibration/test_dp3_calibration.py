"""
Unit tests for dp3 calibration
"""
import importlib

import astropy.units as u
import h5py
import numpy
import pytest
from astropy.coordinates import SkyCoord
from ska_sdp_datamodels.science_data_model.polarisation_model import (
    PolarisationFrame,
)
from ska_sdp_datamodels.sky_model.sky_functions import export_skymodel_to_text
from ska_sdp_datamodels.sky_model.sky_model import SkyComponent, SkyModel

from ska_sdp_func_python.calibration.dp3_calibration import (
    create_parset_from_context,
    dp3_gaincal,
    dp3_gaincal_with_modeldata,
)
from ska_sdp_func_python.sky_model.skymodel_imaging import (
    skymodel_predict_calibrate,
)


@pytest.fixture(autouse=True)
def check_dp3_availability():
    """Check if DP3 is available. If not, the tests in this file are skipped"""
    dp3_loader = importlib.util.find_spec("dp3")
    if dp3_loader is None:
        pytest.skip("DP3 not available")


@pytest.fixture(name="create_skycomponent")
def skycomponent():
    """Create a skycomponent to use for testing"""
    sky_pol_frame = "stokesIQUV"
    frequency = numpy.array([1.0e8])

    f = [100.0, 50.0, -10.0, 40.0]

    flux = numpy.outer(
        numpy.array([numpy.power(freq / 1e8, -0.7) for freq in frequency]),
        f,
    )

    compabsdirection = SkyCoord(
        ra=+181.0 * u.deg, dec=-35.0 * u.deg, frame="icrs", equinox="J2000"
    )
    comp = SkyComponent(
        direction=compabsdirection,
        frequency=frequency,
        flux=flux,
        polarisation_frame=PolarisationFrame(sky_pol_frame),
    )

    return comp


@pytest.fixture(name="create_skycomponent_in_phase_center")
def skycomponent_phase_center():
    """Create a skycomponent in the phase center to use for testing"""
    sky_pol_frame = "stokesIQUV"
    frequency = numpy.array([1.0e8])

    f = [1.0, 0.0, 0.0, 0.0]

    flux = numpy.outer(
        numpy.array([numpy.power(freq / 1e8, -0.7) for freq in frequency]),
        f,
    )

    compabsdirection = SkyCoord(
        ra=+180.0 * u.deg, dec=-35.0 * u.deg, frame="icrs", equinox="J2000"
    )
    comp = SkyComponent(
        direction=compabsdirection,
        frequency=frequency,
        flux=flux,
        polarisation_frame=PolarisationFrame(sky_pol_frame),
    )

    return comp


def test_dp3_gaincal(create_skycomponent, visibility):
    """
    Test that DP3 calibration runs without throwing exception and provides
    the expected result.
    Only run this test if DP3 is available.
    """

    export_skymodel_to_text(
        SkyModel(components=create_skycomponent), "test.skymodel"
    )

    skymodel_vis = skymodel_predict_calibrate(
        visibility, SkyModel(components=create_skycomponent), context="ng"
    )

    dp3_gaincal(skymodel_vis, ["B"], True, solutions_filename="uncorrupted.h5")

    h5_solutions = h5py.File("uncorrupted.h5", "r")
    amplitude_uncorrupted = h5_solutions["sol000/amplitude000/val"][:]

    corruption_factor = 16
    skymodel_vis.vis.data = corruption_factor * skymodel_vis.vis.data
    dp3_gaincal(skymodel_vis, ["B"], True, solutions_filename="corrupted.h5")

    h5_solutions = h5py.File("corrupted.h5", "r")
    amplitude_corrupted = h5_solutions["sol000/amplitude000/val"][:]

    assert numpy.allclose(
        amplitude_corrupted / amplitude_uncorrupted,
        numpy.sqrt(corruption_factor),
        atol=1e-08,
    )


def test_create_parset_from_context(visibility):
    """
    Test that the correct parset is created based on the calibration context.
    Only run this test if DP3 is available.
    """

    calibration_context_list = []
    calibration_context_list.append("T")
    calibration_context_list.append("G")
    calibration_context_list.append("B")

    global_solution = True

    parset_list = create_parset_from_context(
        visibility,
        calibration_context_list,
        global_solution,
        "solutions.h5",
        skymodel_filename="test.skymodel",
    )

    assert len(parset_list) == len(calibration_context_list)

    for i in numpy.arange(len(calibration_context_list)):

        assert parset_list[i].get_string("gaincal.nchan") == "0"
        if calibration_context_list[i] == "T":
            assert (
                parset_list[i].get_string("gaincal.caltype") == "scalarphase"
            )
            assert parset_list[i].get_string("gaincal.solint") == "1"
        elif calibration_context_list[i] == "G":
            assert parset_list[i].get_string("gaincal.caltype") == "diagonal"
            nbins = max(
                1,
                numpy.ceil(
                    (
                        numpy.max(visibility.time.data)
                        - numpy.min(visibility.time.data)
                    )
                    / 60.0
                ).astype("int"),
            )
            assert parset_list[i].get_string("gaincal.solint") == str(nbins)
        elif calibration_context_list[i] == "B":
            assert parset_list[i].get_string("gaincal.caltype") == "diagonal"
            nbins = max(
                1,
                numpy.ceil(
                    (
                        numpy.max(visibility.time.data)
                        - numpy.min(visibility.time.data)
                    )
                    / 1e5
                ).astype("int"),
            )
            assert parset_list[i].get_string("gaincal.solint") == str(nbins)


def test_dp3_gaincal_with_modeldata(
    create_skycomponent_in_phase_center, visibility
):
    """
    Test that DP3 calibration runs without throwing exception when model data
    is provided in a separate visibility object and provides the expected
    result.
    Only run this test if DP3 is available.
    """

    vis = visibility.copy(deep=True)
    skymodel_vis = skymodel_predict_calibrate(
        vis,
        SkyModel(components=create_skycomponent_in_phase_center),
        context="ng",
    )

    corruption_factor = 16
    vis.vis.data = corruption_factor * skymodel_vis.vis.data

    calibrated_vis = dp3_gaincal_with_modeldata(
        vis,
        ["B"],
        True,
        skymodel_vis,
        "model_data",
        solutions_filename="solutions.h5",
    )

    h5_solutions = h5py.File("solutions.h5", "r")
    amplitude_solution = h5_solutions["sol000/amplitude000/val"][:]

    assert numpy.allclose(
        amplitude_solution,
        numpy.sqrt(corruption_factor),
        atol=1e-08,
    )

    assert numpy.allclose(
        calibrated_vis.vis.data,
        skymodel_vis.vis.data,
        atol=1e-08,
    )
