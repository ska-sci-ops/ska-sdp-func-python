"""
Unit tests for image iteration
"""
import logging
import tempfile

import numpy
import pytest
from astropy import units as u
from astropy.coordinates import SkyCoord
from ska_sdp_datamodels.image.image_create import create_image
from ska_sdp_datamodels.science_data_model.polarisation_model import (
    PolarisationFrame,
)
from ska_sdp_datamodels.sky_model.sky_model import SkyComponent

from ska_sdp_func_python.image.iterators import (
    image_channel_iter,
    image_raster_iter,
)
from ska_sdp_func_python.sky_component.operations import insert_skycomponent

log = logging.getLogger("func-python-logger")
log.setLevel(logging.WARNING)

log.setLevel(logging.WARNING)


@pytest.fixture(scope="module", name="input_params")
def iterators_fixture():
    """Fixture for the iterators.py unit tests"""
    phase_centre = SkyCoord(
        ra=+180.0 * u.deg,
        dec=-35.0 * u.deg,
        frame="icrs",
        equinox="J2000",
    )
    image = create_image(
        512,
        0.00015,
        phase_centre,
        nchan=1,
    )

    image_sc = SkyComponent(
        direction=SkyCoord(
            ra=+110.0 * u.deg, dec=-40.0 * u.deg, frame="icrs", equinox="J2000"
        ),
        frequency=numpy.array([1e8]),
        name="image_sc",
        flux=numpy.ones((1, 1)),
        shape="Point",
        polarisation_frame=PolarisationFrame("stokesI"),
    )

    image = insert_skycomponent(image, image_sc)
    params = {
        "image": image,
        "phasecentre": phase_centre,
    }
    return params


def test_raster(input_params):
    """Test a raster iterator across an image. The test is to check that the
    value of the subimages is multiplied by two.

    """

    for npixel in [256, 512, 1024]:
        original = create_image(
            npixel,
            0.00015,
            input_params["phasecentre"],
            nchan=1,
        )
        original["pixels"].data = numpy.ones(
            shape=original["pixels"].data.shape, dtype=float
        )
        assert numpy.max(
            numpy.abs(original["pixels"].data)
        ), "Original is empty"

        for nraster in [1]:
            for overlap in [0, 2, 4, 8, 16]:
                try:
                    model = create_image(
                        npixel,
                        0.00015,
                        input_params["phasecentre"],
                        nchan=1,
                    )
                    model["pixels"].data = numpy.ones(
                        shape=model["pixels"].data.shape, dtype=float
                    )
                    for patch in image_raster_iter(
                        model, facets=nraster, overlap=overlap
                    ):
                        assert patch["pixels"].data.shape[3] == (
                            model["pixels"].data.shape[3] // nraster
                        )
                        assert patch["pixels"].data.shape[2] == (
                            model["pixels"].data.shape[2] // nraster
                        )
                        # Check for frequency and polarisation
                        assert (
                            patch["pixels"].data.shape[0]
                            == model["pixels"].data.shape[0]
                        )
                        assert (
                            patch["pixels"].data.shape[1]
                            == model["pixels"].data.shape[1]
                        )
                        patch["pixels"].data *= 2.0

                    if numpy.max(numpy.abs(model["pixels"].data)) == 0.0:
                        log.warning(
                            "Raster is empty failed for %s, %s, %s",
                            npixel,
                            nraster,
                            overlap,
                        )
                    diff = model.copy(deep=True)
                    diff["pixels"].data -= 2.0 * original["pixels"].data
                    err = numpy.max(diff["pixels"].data)
                    if abs(err) > 0.0:
                        log.warning(
                            "Raster set failed for %s, %s, %s: error %s",
                            npixel,
                            nraster,
                            overlap,
                            err,
                        )
                    with tempfile.TemporaryDirectory() as testdir:
                        model.image_acc.export_to_fits(
                            f"{testdir}/test_image_iterators_model_{npixel}_"
                            f"{nraster}_{overlap}.fits",
                        )
                        diff.image_acc.export_to_fits(
                            f"{testdir}/test_image_iterators_diff_{npixel}_"
                            f"{nraster}_{overlap}.fits",
                        )
                except ValueError as err:
                    log.error(
                        "Iterator failed for %s, %s, %s,: %s",
                        npixel,
                        nraster,
                        overlap,
                        err,
                    )


def test_raster_exception(input_params):
    """Check that raster captures the right exceptions"""
    original = input_params["image"]
    original["pixels"].data = numpy.ones(
        shape=original["pixels"].data.shape, dtype=float
    )
    assert numpy.max(numpy.abs(original["pixels"].data)), "Original is empty"

    for nraster, overlap in [(-1, -1), (-1, 0), (1e6, 127)]:
        with pytest.raises(ValueError):
            model = input_params["image"]
            model["pixels"].data = numpy.ones(
                shape=model["pixels"].data.shape, dtype=float
            )
            for patch in image_raster_iter(
                model, facets=nraster, overlap=overlap
            ):
                patch["pixels"].data *= 2.0

    for nraster, overlap in [(2, 513)]:
        with pytest.raises(ValueError):
            model = input_params["image"]
            for patch in image_raster_iter(
                model, facets=nraster, overlap=overlap
            ):
                patch["pixels"].data *= 2.0


@pytest.mark.skip("Test uses image from file")
def test_channelise():
    """Unit test for the image_channel_iter function"""
    # pylint: disable=E0602
    cube = create_test_image(  # noqa: F821
        frequency=numpy.linspace(1e8, 1.1e8, 128),
        polarisation_frame=PolarisationFrame("stokesI"),
    )

    for subimages in [128, 16, 8, 2, 1]:
        for slab in image_channel_iter(cube, subimages=subimages):
            assert slab["pixels"].data.shape[0] == 128 // subimages
