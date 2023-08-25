"""
Test for SkyModel predict and invert functions
"""
from pathlib import Path

import numpy
import pytest
from ska_sdp_datamodels.sky_model import import_skymodel_from_hdf5

pytest.importorskip(
    modname="ska_sdp_func", reason="ska-sdp-func is an optional dependency"
)
from ska_sdp_func_python.sky_model.skymodel_imaging import (
    dp3_predict,
    skymodel_calibrate_invert,
    skymodel_predict_calibrate,
)
from ska_sdp_func_python.visibility.visibility_geometry import (
    calculate_visibility_parallactic_angles,
)


@pytest.fixture(scope="module", name="low_test_sky_model_from_gleam")
def sky_model_data():
    """
    Sky model loaded from file (see tests/testing_data/README.md)
    """
    sky_model = import_skymodel_from_hdf5(
        str(Path(__file__).parent.absolute())
        + "/../testing_data/low_test_skymodel_from_gleam.hdf"
    )
    return sky_model


def _get_primary_beam(vis, model):
    """
    Create test primary beam
    FIXME: these functions are still in rascil-main
    """
    # pylint: disable=undefined-variable
    pb = create_low_test_beam(model)  # noqa: F821
    pa = numpy.mean(calculate_visibility_parallactic_angles(vis))
    pb = convert_azelvp_to_radec(pb, model, pa)  # noqa: F821
    return pb


def test_predict_calibrate_no_pb(
    visibility_deconv, low_test_sky_model_from_gleam
):
    """Predict with no primary beam"""
    sky_model = low_test_sky_model_from_gleam.copy()

    skymodel_vis = skymodel_predict_calibrate(
        visibility_deconv, sky_model, context="ng"
    )
    qa = skymodel_vis.visibility_acc.qa_visibility()
    numpy.testing.assert_almost_equal(
        qa.data["maxabs"],
        60.35140880932053,
        err_msg=str(qa),
    )


@pytest.mark.skip(reason="TODO: get primary_beam functions")
def test_predict_with_pb(visibility_deconv, low_test_sky_model_from_gleam):
    """Test predict while applying a time-variable primary beam"""
    sky_model = low_test_sky_model_from_gleam.copy()

    skymodel_vis = skymodel_predict_calibrate(
        visibility_deconv,
        sky_model,
        context="ng",
        get_pb=_get_primary_beam,
    )
    qa = skymodel_vis.visibility_acc.qa_visibility()
    numpy.testing.assert_almost_equal(
        qa.data["maxabs"],
        32.20530966848842,
        err_msg=str(qa),
    )


def test_calibrate_invert_no_pb(
    visibility_deconv, low_test_sky_model_from_gleam
):
    """Test invert without primary beam"""
    sky_model = low_test_sky_model_from_gleam.copy()

    skymodel_vis = skymodel_predict_calibrate(
        visibility_deconv,
        sky_model,
        context="ng",
    )
    assert numpy.max(numpy.abs(skymodel_vis.vis)) > 0.0

    dirty, _ = skymodel_calibrate_invert(
        skymodel_vis,
        sky_model,
        normalise=True,
        flat_sky=False,
    )
    qa = dirty.image_acc.qa_image()

    numpy.testing.assert_allclose(
        qa.data["max"],
        4.196352244614598,
        atol=1e-7,
        err_msg=f"{qa}",
    )
    numpy.testing.assert_allclose(
        qa.data["min"],
        -0.3160421749599381,
        atol=1e-7,
        err_msg=f"{qa}",
    )


@pytest.mark.skip(reason="TODO: get primary_beam functions")
def test_invert_with_pb(visibility_deconv, low_test_sky_model_from_gleam):
    """Test invert while applying a time-variable primary beam"""
    sky_model = low_test_sky_model_from_gleam.copy()
    skymodel_vis = skymodel_predict_calibrate(
        visibility_deconv,
        sky_model,
        context="ng",
        get_pb=_get_primary_beam,
    )
    assert numpy.max(numpy.abs(skymodel_vis.vis)) > 0.0

    skymodel = skymodel_calibrate_invert(
        skymodel_vis,
        sky_model,
        get_pb=_get_primary_beam,
        normalise=True,
        flat_sky=False,
    )

    qa = skymodel[0].image_acc.qa_image()

    numpy.testing.assert_allclose(
        qa.data["max"], 3.767454977596991, atol=1e-7, err_msg=f"{qa}"
    )
    numpy.testing.assert_allclose(
        qa.data["min"], -0.23958139130004705, atol=1e-7, err_msg=f"{qa}"
    )

    # Now repeat with flat_sky=True
    skymodel = skymodel_calibrate_invert(
        skymodel_vis,
        skymodel,
        get_pb=_get_primary_beam,
        normalise=True,
        flat_sky=True,
    )

    qa = skymodel[0].image_acc.qa_image()

    numpy.testing.assert_allclose(
        qa.data["max"],
        4.025153684707801,
        atol=1e-7,
        err_msg=f"{qa}",
    )
    numpy.testing.assert_allclose(
        qa.data["min"],
        -0.24826345131847594,
        atol=1e-7,
        err_msg=f"{qa}",
    )


def test_predict_no_components(
    visibility_deconv, low_test_sky_model_from_gleam
):
    """Test predict with no components"""
    sky_model = low_test_sky_model_from_gleam.copy()
    sky_model.components = []

    assert (
        numpy.max(numpy.abs(sky_model.image["pixels"].data)) > 0.0
    ), "Image is empty"

    skymodel_vis = skymodel_predict_calibrate(
        visibility_deconv, sky_model, context="ng"
    )
    qa = skymodel_vis.visibility_acc.qa_visibility()
    numpy.testing.assert_almost_equal(
        qa.data["maxabs"], 39.916746503252156, err_msg=str(qa)
    )


def test_predict_no_image(visibility_deconv, low_test_sky_model_from_gleam):
    """Test predict with no image"""
    sky_model = low_test_sky_model_from_gleam.copy()
    sky_model.image = None

    skymodel_vis = skymodel_predict_calibrate(
        visibility_deconv, sky_model, context="ng"
    )
    qa = skymodel_vis.visibility_acc.qa_visibility()
    numpy.testing.assert_almost_equal(
        qa.data["maxabs"],
        20.434662306068372,
        err_msg=str(qa),
    )


def test_dp3_predict(visibility_deconv, low_test_sky_model_from_gleam):
    """Test DP3 predict"""
    sky_model = low_test_sky_model_from_gleam.copy()

    skymodel_vis = dp3_predict(
        visibility_deconv,
        sky_model,
    )
    assert numpy.max(numpy.abs(skymodel_vis.vis)) > 0.0


def test_dp3_predict_kwargs(visibility_deconv, low_test_sky_model_from_gleam):
    """Test DP3 predict with extra arguments"""
    sky_model = low_test_sky_model_from_gleam.copy()

    dp3_predict(visibility_deconv, sky_model, correctfreqsmearing="True")

    with pytest.raises(ValueError) as e:
        dp3_predict(visibility_deconv, sky_model, dummy_setting="skymodel")
    assert "DP3 predict does not support the argument: dummy_setting" in str(
        e.value
    )
