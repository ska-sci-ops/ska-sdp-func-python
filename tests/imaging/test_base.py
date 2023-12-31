"""
Unit tests for base imaging functions
"""
import numpy
import pytest

from ska_sdp_func_python.imaging.base import (
    advise_wide_field,
    create_image_from_visibility,
    fill_vis_for_psf,
    invert_awprojection,
    predict_awprojection,
    shift_vis_to_image,
    visibility_recentre,
)


def test_shift_vis_to_image(visibility, image, comp_direction):
    """
    Unit tests for shift_vis_to_image function:
    check that the phasecentre does change
    """
    vis = visibility.copy(deep=True)
    old_pc = visibility.attrs["phasecentre"]
    shifted_vis = shift_vis_to_image(vis, image)

    assert old_pc != shifted_vis.attrs["phasecentre"]
    assert shifted_vis.attrs["phasecentre"] == comp_direction


@pytest.mark.skip(reason="gcfcf examples needed for predict_awprojection")
def test_predict_awprojection(visibility, image):
    """
    Test predict_awprojection
    """
    vis = visibility.copy(deep=True)
    svis = predict_awprojection(
        vis,
        image,
    )

    assert visibility != svis


def test_fill_vis_for_psf(visibility):
    """
    Unit tests for fill_vis_for_psf function
    Note: visibility polarisation is linear
    """
    vis = visibility.copy(deep=True)
    assert (vis["vis"].data[...] == 0.0 + 0.0j).all()

    result = fill_vis_for_psf(vis)
    assert (result["vis"].data[..., 0] == 1.0 + 0.0j).all()
    assert (result["vis"].data[..., 1:3] == 0.0 + 0.0j).all()
    assert (result["vis"].data[..., 3] == 1.0 + 0.0j).all()


def test_create_image_from_visibility(visibility, image, comp_direction):
    """
    Unit tests for create_image_from_visibility function

    The inputs to creating the image fixture come from the
    visibility fixture with the difference in the phase_centre
    which for the image is the "comp_direction" fixture
    """
    new_image = create_image_from_visibility(
        vis=visibility,
        phasecentre=comp_direction,
    )

    assert (new_image == image).all()


@pytest.mark.skip(reason="Need more info on gcfcf values")
def test_invert_awprojection(visibility, image):
    """Unit tests for normalise_sumwt function:
    check image created here is the same as image in result_base
    """
    inverted_im = invert_awprojection(visibility, image, gcfcf="")

    assert inverted_im != image


def test_visibility_recentre():
    """Unit tests for normalise_sumwt function:
    check image created here is the same as image in result_base
    """
    uvw = numpy.array([1, 2, 3])
    dl = 0.1
    dm = 0.5
    uvw_recentred = visibility_recentre(uvw, dl, dm)
    assert uvw_recentred[0] == 0.7
    assert uvw_recentred[1] == 0.5
    assert uvw_recentred[2] == 3


def test_advise_wide_field(visibility):
    """
    Unit test for advise_wide_field
    Use default parameters for advising
    """
    advice = advise_wide_field(visibility)

    # All entries in the dictionary are provided
    assert len(advice) == 35
    for key in advice:
        assert advice[key] is not None

    # Some key entries are in expected value
    assert advice["delA"] == 0.02
    assert advice["oversampling_synthesised_beam"] == 3.0
    # npixels2 directly depends on the baselines
    # the input visibility uses the full LOW array
    assert advice["npixels2"] == 65536
    assert advice["wstep_primary_beam"] == advice["w_sampling_primary_beam"]
    assert advice["wprojection_planes_image"] == advice["vis_slices_image"]
