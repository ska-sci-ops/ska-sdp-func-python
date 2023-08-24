# pylint: skip-file
"""
Unit tests for imaging using nifty gridder
"""
import logging
import os
import sys
import tempfile

import numpy
import pytest
from astropy import units as u
from astropy.coordinates import SkyCoord
from ska_sdp_datamodels.configuration.config_create import (
    create_named_configuration,
)
from ska_sdp_datamodels.image.image_create import create_image
from ska_sdp_datamodels.science_data_model.polarisation_model import (
    PolarisationFrame,
)
from ska_sdp_datamodels.visibility.vis_create import create_visibility

from ska_sdp_func_python.imaging.wg import invert_wg, predict_wg

pytest.skip(allow_module_level=True, reason="Imports for WAGG")

log = logging.getLogger("func-python-logger")

log.setLevel(logging.WARNING)
log.addHandler(logging.StreamHandler(sys.stdout))


@pytest.fixture(scope="module", name="result_wg")
def wg_fixture():
    persist = os.getenv("FUNC_PYTHON_PERSIST", False)
    verbosity = 0
    npixel = 256
    low = create_named_configuration("LOWBD2", rmax=750.0)
    ntimes = 5
    times = numpy.linspace(-3.0, +3.0, ntimes) * numpy.pi / 12.0
    frequency = numpy.array([1e8])
    channelwidth = numpy.array([1e6])

    vis_pol = PolarisationFrame("stokesI")

    phase_centre = SkyCoord(
        ra=+180.0 * u.deg, dec=-45.0 * u.deg, frame="icrs", equinox="J2000"
    )
    vis = create_visibility(
        low,
        times,
        frequency,
        phase_centre,
        channelwidth,
        polarisation_frame=vis_pol,
    )

    model = create_image(
        npixel,
        0.00015,
        phase_centre,
        nchan=1,
    )

    params = {
        "model": model,
        "persist": persist,
        "verbosity": verbosity,
        "visibility": vis,
    }
    return params


def test_predict_wg(result_wg):
    vis = result_wg["visibility"]
    model = result_wg["model"]
    verbosity = result_wg["verbosity"]
    persist = result_wg["persist"]

    original_vis = vis.copy(deep=True)
    vis = predict_wg(vis, model, verbosity=verbosity)
    vis["vis"].data = vis["vis"].data - original_vis["vis"].data
    dirty = invert_wg(
        vis,
        model,
        dopsf=False,
        normalise=True,
        verbosity=verbosity,
    )

    if persist:
        with tempfile.TemporaryDirectory() as tempdir:
            dirty[0].image_acc.export_to_fits(
                f"{tempdir}/test_imaging_ng_predict_wg_residual.fits"
            )

    maxabs = numpy.max(numpy.abs(dirty[0]["pixels"].data))
    assert maxabs < 1, "Error %.3f greater than fluxthreshold %.3f " % (
        maxabs,
        1,
    )


# def test_invert_wg(result_wg):
#     vis = result_wg["visibility"]
#     model = result_wg["model"]
#     verbosity = result_wg["verbosity"]
#     persist = result_wg["persist"]
#     dirty = invert_wg(
#         vis,
#         model,
#         normalise=True,
#         verbosity=verbosity,
#     )
#
#     if persist:
#         with tempfile.TemporaryDirectory() as tempdir:
#             dirty[0].image_acc.export_to_fits(
#                 f"{tempdir}/test_imaging_ng_invert_wg_dirty.fits"
#             )
#     assert numpy.max(numpy.abs(dirty[0]["pixels"].data)), "Image is empty"
#
#
# def test_invert_wg_psf(result_wg):
#     vis = result_wg["visibility"]
#     model = result_wg["model"]
#     verbosity = result_wg["verbosity"]
#     persist = result_wg["persist"]
#     dirty = invert_wg(
#         vis,
#         model,
#         normalise=True,
#         dopsf=True,
#         verbosity=verbosity,
#     )
#
#     if persist:
#         with tempfile.TemporaryDirectory() as tempdir:
#             dirty[0].image_acc.export_to_fits(
#                 f"{tempdir}/test_imaging_ng_invert_wg_psf_dirty.fits"
#             )
#     assert numpy.max(numpy.abs(dirty[0]["pixels"].data)), "Image is empty"
