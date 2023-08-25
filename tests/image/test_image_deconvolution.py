"""
Unit tests for image deconvolution
"""
import numpy
import pytest
from ska_sdp_datamodels.image.image_create import create_image
from ska_sdp_datamodels.science_data_model.polarisation_model import (
    PolarisationFrame,
)

from ska_sdp_func_python.image.cleaners import overlapIndices
from ska_sdp_func_python.image.deconvolution import (
    deconvolve_cube,
    find_window_list,
    fit_psf,
    hogbom_kernel_list,
    restore_cube,
    restore_list,
)
from ska_sdp_func_python.imaging.base import create_image_from_visibility
from ska_sdp_func_python.imaging.imaging import (
    invert_visibility,
    predict_visibility,
)


@pytest.fixture(scope="module", name="predicted_vis")
def predicted_vis_fixture(visibility_deconv):
    """Pytest fixture for the deconvolution.py unit tests"""
    # pylint: disable=import-outside-toplevel
    from numpy.random import default_rng

    rng = default_rng(1805550721)

    vis = visibility_deconv.copy(deep=True, zero=True)
    test_model = create_image(
        256,
        0.001,
        vis.phasecentre,
        frequency=visibility_deconv.frequency[0],
        channel_bandwidth=visibility_deconv.channel_bandwidth[0],
        nchan=1,
    )
    test_model["pixels"].data[:, :, 31:209, 33:224] = rng.normal(
        0.1018, 0.1241, (1, 1, 178, 191)
    )  # trying roughly to replicate loading data from an M31 fits image
    test_model["pixels"].data[...] = test_model["pixels"].data[...]
    vis = predict_visibility(vis, test_model, context="2d")
    assert numpy.max(numpy.abs(vis.vis)) > 0.0
    return vis


@pytest.fixture(scope="module", name="model")
def model_fixture(predicted_vis):
    """Fixture of model created from predicted_vis"""
    model = create_image_from_visibility(
        predicted_vis,
        npixel=512,
        cellsize=0.001,
        polarisation_frame=PolarisationFrame("stokesI"),
    )
    return model


@pytest.fixture(scope="module", name="dirty_img")
def dirty_img_fixture(predicted_vis, model):
    """Fixture of dirty image"""
    dirty = invert_visibility(predicted_vis, model, context="2d")[0]
    return dirty


@pytest.fixture(scope="module", name="psf")
def psf_fixture(predicted_vis, model):
    """Fixture of point-spread function (PSF)"""
    psf = invert_visibility(predicted_vis, model, context="2d", dopsf=True)[0]
    return psf


# TODO: create_pb is still in rascil-main
@pytest.fixture(scope="module", name="sensitivity")
def sensitivity_fixture(model):
    """Fixture of sensitivity image"""
    # pylint: disable=undefined-variable
    sensitivity = create_pb(model, "LOW")  # noqa: F821
    return sensitivity


def check_overlap(a1, a2, s1, s2):
    """Check overlaps"""
    a1[s1[0] : s1[1], s1[2] : s1[3]] = 1
    a2[s2[0] : s2[1], s2[2] : s2[3]] = 1
    return numpy.sum(a1) == numpy.sum(a2)


def test_overlap():
    """Unit tests for the overlapIndices function"""
    res = numpy.zeros([512, 512])
    psf = numpy.zeros([100, 100])
    peak = (499, 249)
    s1, s2 = overlapIndices(res, psf, peak[0], peak[1])
    assert len(s1) == 4
    assert len(s2) == 4
    check_overlap(res, psf, s1, s2)
    assert s1 == (449, 512, 199, 299)
    assert s2 == (0, 63, 0, 100)


def test_restore(model, psf):
    """Unit tests for the restore_cube function"""
    model.data_vars["pixels"].data[0, 0, 256, 256] = 1.0
    cmodel = restore_cube(model, psf)
    assert numpy.abs(numpy.max(cmodel["pixels"].data) - 1.0) < 1e-7, numpy.max(
        cmodel["pixels"].data
    )


def test_restore_list(model, psf):
    """Unit tests for the restore_list function"""
    model["pixels"].data[0, 0, 256, 256] = 1.0
    cmodel = restore_list([model], [psf])[0]
    assert numpy.abs(numpy.max(cmodel["pixels"].data) - 1.0) < 1e-7, numpy.max(
        cmodel["pixels"].data
    )


def test_restore_clean_beam(model, psf):
    """Test restoration with specified beam

    :return:
    """
    model["pixels"].data[0, 0, 256, 256] = 1.0
    # The beam is specified in degrees
    bmaj = 0.006 * 180.0 / numpy.pi
    cmodel = restore_cube(
        model,
        psf,
        clean_beam={"bmaj": bmaj, "bmin": bmaj, "bpa": 0.0},
    )
    assert numpy.abs(numpy.max(cmodel["pixels"].data) - 1.0) < 1e-7, numpy.max(
        cmodel["pixels"].data
    )


def test_fit_psf(psf):
    """Unit tests for the fit_psf function"""
    clean_beam = fit_psf(psf)

    assert numpy.abs(clean_beam["bmaj"] - 0.3232026716040641) < 6.0e-6
    assert numpy.abs(clean_beam["bmin"] - 0.2773962393170226) < 6.0e-6


def test_deconvolve_hogbom(dirty_img, psf):
    """Unit tests for the deconvolve_cube function using hogbom"""
    _, residual = deconvolve_cube(
        dirty_img,
        psf,
        niter=10000,
        gain=0.1,
        algorithm="hogbom",
        threshold=0.01,
    )
    print(numpy.max(residual["pixels"].data))
    assert numpy.max(residual["pixels"].data) < 0.4


def test_deconvolve_msclean(dirty_img, psf):
    """Unit tests for the deconvolve_cube function using msclean"""
    _, residual = deconvolve_cube(
        dirty_img,
        psf,
        niter=1000,
        gain=0.7,
        algorithm="msclean",
        scales=[0, 3, 10, 30],
        threshold=0.01,
    )
    assert numpy.max(residual["pixels"].data) < 1.3


def test_deconvolve_msclean_1scale(dirty_img, psf):
    """Unit tests for the deconvolve_cube function using msclean and scale 1"""
    _, residual = deconvolve_cube(
        dirty_img,
        psf,
        niter=10000,
        gain=0.1,
        algorithm="msclean",
        scales=[0],
        threshold=0.01,
    )
    assert numpy.max(residual["pixels"].data) < 0.4


def test_deconvolve_hogbom_no_edge(dirty_img, psf):
    """Unit tests for the deconvolve_cube function using hogbom and no_edge"""
    _, residual = deconvolve_cube(
        dirty_img,
        psf,
        window_shape="no_edge",
        niter=10000,
        gain=0.1,
        algorithm="hogbom",
        threshold=0.01,
    )
    assert numpy.max(residual["pixels"].data) < 0.7


def test_deconvolve_hogbom_inner_quarter(dirty_img, psf):
    """Unit tests for the deconvolve_cube function using hogbom and quarter"""
    _, residual = deconvolve_cube(
        dirty_img,
        psf,
        window_shape="quarter",
        niter=10000,
        gain=0.1,
        algorithm="hogbom",
        threshold=0.01,
    )
    assert numpy.max(residual["pixels"].data) < 0.7


def test_deconvolve_msclean_inner_quarter(dirty_img, psf):
    """Unit tests for the deconvolve_cube function using msclean and quarter"""
    _, residual = deconvolve_cube(
        dirty_img,
        psf,
        window_shape="quarter",
        niter=1000,
        gain=0.7,
        algorithm="msclean",
        scales=[0, 3, 10, 30],
        threshold=0.01,
    )
    assert numpy.max(residual["pixels"].data) < 0.8


def test_deconvolve_hogbom_subpsf(dirty_img, psf):
    """Unit tests for the deconvolve_cube function"""
    _, residual = deconvolve_cube(
        dirty_img,
        psf,
        psf_support=200,
        window_shape="quarter",
        niter=10000,
        gain=0.1,
        algorithm="hogbom",
        threshold=0.01,
    )
    assert (
        numpy.max(residual["pixels"].data[..., 56:456, 56:456])
        < 0.7713111251174253
    )


def test_deconvolve_msclean_subpsf(dirty_img, psf):
    """Unit tests for the deconvolve_cube function"""
    _, residual = deconvolve_cube(
        dirty_img,
        psf,
        psf_support=200,
        window_shape="quarter",
        niter=1000,
        gain=0.7,
        algorithm="msclean",
        scales=[0, 3, 10, 30],
        threshold=0.01,
    )
    assert (
        numpy.max(residual["pixels"].data[..., 56:456, 56:456])
        < 0.9147349758258286
    )


@pytest.mark.skip("Sensitivity function not available yet (primary_beam func)")
def test_deconvolve_msclean_sensitivity(dirty_img, psf, sensitivity):
    """Test deconvolve_cube using a sensitibity image"""
    _, residual = deconvolve_cube(
        dirty_img,
        psf,
        sensitivity=sensitivity,
        niter=1000,
        gain=0.7,
        algorithm="msclean",
        scales=[0, 3, 10, 30],
        threshold=0.01,
    )

    qa = residual.image_acc.qa_image()
    numpy.testing.assert_allclose(
        qa.data["max"], 0.8040729590477751, atol=1e-7, err_msg=f"{qa}"
    )
    numpy.testing.assert_allclose(
        qa.data["min"], -0.9044553283128349, atol=1e-7, err_msg=f"{qa}"
    )


def _check_hogbom_kernel_list_test_results(component, residual):
    """Test non zero values"""
    result_comp_data = component["pixels"].data
    non_zero_idx_comp = numpy.where(result_comp_data != 0.0)
    expected_comp_non_zero_data = numpy.array(
        [
            0.23512022,
            0.24861933,
            0.27351581,
            0.3005493,
            0.22978201,
            0.23196268,
            0.22754817,
            0.34480517,
            0.24560397,
            0.33851125,
        ]
    )
    result_residual_data = residual["pixels"].data
    non_zero_idx_residual = numpy.where(result_residual_data != 0.0)
    expected_residual_non_zero_data = numpy.array(
        [
            -0.06446651,
            -0.20050621,
            -0.25780698,
            -0.23257206,
            -0.15831075,
            -0.10916665,
            -0.11743437,
            -0.16703326,
            -0.22206081,
            -0.26290388,
        ]
    )

    # number of non-zero values
    assert len(result_comp_data[non_zero_idx_comp]) == 86
    assert len(result_residual_data[non_zero_idx_residual]) == 262144
    # test first 10 non-zero values don't change with each run of test
    numpy.testing.assert_array_almost_equal(
        result_comp_data[non_zero_idx_comp][:10],
        expected_comp_non_zero_data,
    )
    numpy.testing.assert_array_almost_equal(
        result_residual_data[non_zero_idx_residual][:10],
        expected_residual_non_zero_data,
    )


def test_hogbom_kernel_list_single_dirty(dirty_img, psf):
    """Unit tests for the find_window_list function"""
    prefix = "test_hogbom_list"
    dirty_list = [dirty_img]
    psf_list = [psf]
    window_list = find_window_list(dirty_list, prefix)

    comp_list, residual_list = hogbom_kernel_list(
        dirty_list, prefix, psf_list, window_list
    )

    assert len(comp_list) == 1
    assert len(residual_list) == 1
    _check_hogbom_kernel_list_test_results(comp_list[0], residual_list[0])


def test_hogbom_kernel_list_multiple_dirty(dirty_img, psf):
    """
    Bugfix: hogbom_kernel_list produced an IndexError, when
    dirty_list has more than one elements, and those elements are
    for a single frequency each, and window_shape is None.
    """

    prefix = "test_hogbom_list"
    dirty_list = [dirty_img, dirty_img]
    psf_list = [psf, psf]
    window_list = find_window_list(dirty_list, prefix, window_shape=None)

    comp_list, residual_list = hogbom_kernel_list(
        dirty_list, prefix, psf_list, window_list
    )

    assert len(comp_list) == 2
    assert len(residual_list) == 2
    # because the two dirty images and psfs are the same,
    # the expected results are also the same
    _check_hogbom_kernel_list_test_results(comp_list[0], residual_list[0])
    _check_hogbom_kernel_list_test_results(comp_list[1], residual_list[1])


def test_hogbom_kernel_list_multiple_dirty_window_shape(dirty_img, psf):
    """
    Bugfix: hogbom_kernel_list produced an IndexError.
    Test the second branch of the if statement
    when dirty_list has more than one elements.
    """
    prefix = "test_hogbom_list"
    dirty_list = [dirty_img, dirty_img]
    psf_list = [psf, psf]
    window_list = find_window_list(dirty_list, prefix, window_shape="quarter")

    comp_list, residual_list = hogbom_kernel_list(
        dirty_list, prefix, psf_list, window_list
    )

    assert len(comp_list) == 2
    assert len(residual_list) == 2
    # because the two dirty images and psfs are the same,
    # the expected results are also the same
    _check_hogbom_kernel_list_test_results(comp_list[0], residual_list[0])
    _check_hogbom_kernel_list_test_results(comp_list[1], residual_list[1])
