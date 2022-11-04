# pylint: disable=invalid-name, too-many-arguments
# pylint: disable=too-many-instance-attributes, invalid-envvar-default
# pylint: disable=missing-class-docstring, missing-function-docstring
# pylint: disable=consider-using-f-string, unused-variable
# pylint: disable=attribute-defined-outside-init, too-many-public-methods
# pylint: disable=import-error, no-name-in-module, import-outside-toplevel
""" Unit tests for image operations


"""
import pytest

pytestmark = pytest.skip(allow_module_level=True)
import functools
import logging
import os
import sys
import tempfile
import unittest

import numpy
from astropy import units as u
from astropy.coordinates import SkyCoord
from ska_sdp_datamodels.science_data_model.polarisation_model import (
    PolarisationFrame,
)

from ska_sdp_func_python.griddata.gridding import (
    degrid_visibility_from_griddata,
    fft_griddata_to_image,
    fft_image_to_griddata,
    grid_visibility_to_griddata,
    grid_visibility_weight_to_griddata,
    griddata_merge_weights,
    griddata_visibility_reweight,
)
from ska_sdp_func_python.image.operations import (
    convert_polimage_to_stokes,
    convert_stokes_to_polimage,
)
from ska_sdp_func_python.imaging.base import normalise_sumwt
from ska_sdp_func_python.imaging.dft import dft_skycomponent_visibility
from ska_sdp_func_python.skycomponent.operations import insert_skycomponent

# fix the below imports
from src.ska_sdp_func_python.griddata.kernels import (
    create_awterm_convolutionfunction,
    create_box_convolutionfunction,
    create_pswf_convolutionfunction,
)
from src.ska_sdp_func_python.griddata.operations import (
    create_griddata_from_image,
)
from src.ska_sdp_func_python.image.operations import smooth_image
from src.ska_sdp_func_python.imaging.primary_beams import create_pb_generic
from src.ska_sdp_func_python.simulation import (
    create_named_configuration,
    create_unittest_components,
    create_unittest_model,
    ingest_unittest_visibility,
)

log = logging.getLogger("func-python-logger")

log.setLevel(logging.WARNING)
log.addHandler(logging.StreamHandler(sys.stdout))


class TestGridDataGridding(unittest.TestCase):
    def setUp(self):
        self.persist = os.getenv("FUNC_PYTHON_PERSIST", False)

    def actualSetUp(
        self,
        zerow=True,
        image_pol=PolarisationFrame("stokesIQUV"),
        test_ignored_visibilities=False,
    ):

        self.doplot = False
        self.npixel = 256
        self.cellsize = 0.0009
        self.low = create_named_configuration("LOWBD2", rmax=750.0)
        self.freqwin = 1
        self.vis_list = []
        self.ntimes = 3
        self.times = numpy.linspace(-2.0, +2.0, self.ntimes) * numpy.pi / 12.0

        if self.freqwin == 1:
            self.frequency = numpy.array([1e8])
            self.channelwidth = numpy.array([4e7])
        else:
            self.frequency = numpy.linspace(0.8e8, 1.2e8, self.freqwin)
            self.channelwidth = numpy.array(
                self.freqwin * [self.frequency[1] - self.frequency[0]]
            )

        self.image_pol = image_pol
        if image_pol == PolarisationFrame("stokesI"):
            self.vis_pol = PolarisationFrame("stokesI")
            f = numpy.array([100.0])
        elif image_pol == PolarisationFrame("stokesIQUV"):
            self.vis_pol = PolarisationFrame("linear")
            f = numpy.array([100.0, 20.0, -10.0, 1.0])
        elif image_pol == PolarisationFrame("stokesIQ"):
            self.vis_pol = PolarisationFrame("linearnp")
            f = numpy.array([100.0, 20.0])
        elif image_pol == PolarisationFrame("stokesIV"):
            self.vis_pol = PolarisationFrame("circularnp")
            f = numpy.array([100.0, 20.0])
        else:
            raise ValueError("Polarisation {} not supported".format(image_pol))

        flux = numpy.array(
            [f * numpy.power(freq / 1e8, -0.7) for freq in self.frequency]
        )

        self.phasecentre = SkyCoord(
            ra=+180.0 * u.deg, dec=-60.0 * u.deg, frame="icrs", equinox="J2000"
        )
        self.vis = ingest_unittest_visibility(
            self.low,
            self.frequency,
            self.channelwidth,
            self.times,
            self.vis_pol,
            self.phasecentre,
            zerow=zerow,
        )
        if test_ignored_visibilities:
            self.cellsize = 1 / (
                2
                * numpy.min(self.vis.visibility_acc.uvw_lambda[..., 0, 0].flat)
            )

        self.model = create_unittest_model(
            self.vis,
            self.image_pol,
            cellsize=self.cellsize,
            npixel=self.npixel,
            nchan=self.freqwin,
        )
        self.components = create_unittest_components(
            self.model,
            flux,
            applypb=False,
            scale=0.5,
            single=False,
            symmetric=True,
        )
        self.model = insert_skycomponent(self.model, self.components)

        self.vis = dft_skycomponent_visibility(self.vis, self.components)

        # Calculate the model convolved with a Gaussian.
        self.cmodel = smooth_image(self.model)
        if self.persist:
            with tempfile.TemporaryDirectory() as tempdir:
                self.model.image_acc.export_to_fits(
                    f"{tempdir}/test_gridding_model.fits"
                )
                self.cmodel.image_acc.export_to_fits(
                    f"{tempdir}/test_gridding_cmodel.fits"
                )
        pb = create_pb_generic(
            self.model, diameter=35.0, blockage=0.0, use_local=False
        )
        self.cmodel["pixels"].data *= pb["pixels"].data
        if self.persist:
            with tempfile.TemporaryDirectory() as tempdir:
                self.cmodel.image_acc.export_to_fits(
                    f"{tempdir}/test_gridding_cmodel_pb.fits"
                )
        self.peak = numpy.unravel_index(
            numpy.argmax(numpy.abs(self.cmodel["pixels"].data)),
            self.cmodel["pixels"].shape,
        )

    def test_griddata_invert_pswf(self):
        self.actualSetUp(zerow=True)
        gcf, cf = create_pswf_convolutionfunction(
            self.model, polarisation_frame=self.vis_pol
        )
        griddata = create_griddata_from_image(
            self.model, polarisation_frame=self.vis_pol
        )
        griddata, sumwt = grid_visibility_to_griddata(
            self.vis, griddata=griddata, cf=cf
        )
        cim = fft_griddata_to_image(griddata, gcf, gcf)
        cim = normalise_sumwt(cim, sumwt)
        im = convert_polimage_to_stokes(cim)
        if self.persist:
            with tempfile.TemporaryDirectory() as tempdir:
                im.image_acc.export_to_fits(
                    f"{tempdir}/test_gridding_dirty_pswf.fits"
                )
        self.check_peaks(im, 97.10594988, tol=1e-7)

    def test_griddata_invert_pswf_stokesIQ(self):
        self.actualSetUp(zerow=True, image_pol=PolarisationFrame("stokesIQ"))
        gcf, cf = create_pswf_convolutionfunction(
            self.model, polarisation_frame=self.vis_pol
        )
        griddata = create_griddata_from_image(
            self.model, polarisation_frame=self.vis_pol
        )
        griddata, sumwt = grid_visibility_to_griddata(
            self.vis, griddata=griddata, cf=cf
        )
        cim = fft_griddata_to_image(griddata, gcf, gcf)
        cim = normalise_sumwt(cim, sumwt)
        im = convert_polimage_to_stokes(cim)
        if self.persist:
            with tempfile.TemporaryDirectory() as tempdir:
                im.image_acc.export_to_fits(
                    f"{tempdir}/test_gridding_dirty_pswf.fits"
                )
        self.check_peaks(im, 97.10594988, tol=1e-7)

    def test_griddata_invert_aterm(self):
        self.actualSetUp(zerow=True)
        make_pb = functools.partial(
            create_pb_generic, diameter=35.0, blockage=0.0, use_local=False
        )
        pb = make_pb(self.model)
        if self.persist:
            with tempfile.TemporaryDirectory() as tempdir:
                pb.image_acc.export_to_fits(
                    f"{tempdir}/test_gridding_aterm_pb.fits"
                )
        gcf, cf = create_awterm_convolutionfunction(
            self.model,
            make_pb=make_pb,
            nw=1,
            oversampling=16,
            support=16,
            use_aaf=False,
            polarisation_frame=self.vis_pol,
        )

        griddata = create_griddata_from_image(
            self.model, polarisation_frame=self.vis_pol
        )
        griddata, sumwt = grid_visibility_to_griddata(
            self.vis, griddata=griddata, cf=cf
        )
        cim = fft_griddata_to_image(griddata, gcf, gcf)
        cim = normalise_sumwt(cim, sumwt)
        im = convert_polimage_to_stokes(cim)
        if self.persist:
            with tempfile.TemporaryDirectory() as tempdir:
                im.image_acc.export_to_fits(
                    f"{tempdir}/test_gridding_dirty_aterm.fits"
                )
        self.check_peaks(im, 97.10594988, tol=1e-7)

    def test_griddata_invert_aterm_noover(self):
        self.actualSetUp(zerow=True)
        make_pb = functools.partial(
            create_pb_generic, diameter=35.0, blockage=0.0, use_local=False
        )
        pb = make_pb(self.model)
        if self.persist:
            with tempfile.TemporaryDirectory() as tempdir:
                pb.image_acc.export_to_fits(
                    f"{tempdir}/test_gridding_aterm_pb.fits"
                )
        gcf, cf = create_awterm_convolutionfunction(
            self.model,
            make_pb=make_pb,
            nw=1,
            oversampling=1,
            support=16,
            use_aaf=True,
            polarisation_frame=self.vis_pol,
        )
        griddata = create_griddata_from_image(
            self.model, polarisation_frame=self.vis_pol
        )
        griddata, sumwt = grid_visibility_to_griddata(
            self.vis, griddata=griddata, cf=cf
        )
        cim = fft_griddata_to_image(griddata, gcf, gcf)
        cim = normalise_sumwt(cim, sumwt)
        im = convert_polimage_to_stokes(cim)
        if self.persist:
            with tempfile.TemporaryDirectory() as tempdir:
                im.image_acc.export_to_fits(
                    f"{tempdir}/test_gridding_dirty_aterm_noover.fits"
                )
        self.check_peaks(im, 97.10594988489598)

    def test_griddata_invert_box(self):
        self.actualSetUp(zerow=True)
        gcf, cf = create_box_convolutionfunction(
            self.model, polarisation_frame=self.vis_pol
        )
        griddata = create_griddata_from_image(
            self.model, polarisation_frame=self.vis_pol
        )
        griddata, sumwt = grid_visibility_to_griddata(
            self.vis, griddata=griddata, cf=cf
        )
        cim = fft_griddata_to_image(griddata, gcf, gcf)
        cim = normalise_sumwt(cim, sumwt)
        im = convert_polimage_to_stokes(cim)
        if self.persist:
            with tempfile.TemporaryDirectory() as tempdir:
                im.image_acc.export_to_fits(
                    f"{tempdir}/test_gridding_dirty_box.fits"
                )
        self.check_peaks(im, 97.10594988489598, tol=1e-7)

    def check_peaks(self, im, peak, tol=1e-6):
        assert numpy.abs(im["pixels"].data[self.peak] - peak) < tol, im[
            "pixels"
        ].data[self.peak]

    def test_griddata_invert_wterm(self):
        self.actualSetUp(zerow=False)
        gcf, cf = create_awterm_convolutionfunction(
            self.model,
            nw=100,
            wstep=8.0,
            oversampling=4,
            support=32,
            use_aaf=True,
            polarisation_frame=self.vis_pol,
        )

        griddata = create_griddata_from_image(
            self.model, polarisation_frame=self.vis_pol
        )
        griddata, sumwt = grid_visibility_to_griddata(
            self.vis, griddata=griddata, cf=cf
        )
        cim = fft_griddata_to_image(griddata, gcf, gcf)
        cim = normalise_sumwt(cim, sumwt)
        im = convert_polimage_to_stokes(cim)
        if self.persist:
            with tempfile.TemporaryDirectory() as tempdir:
                im.image_acc.export_to_fits(
                    f"{tempdir}/test_gridding_dirty_wterm.fits"
                )
        self.check_peaks(im, 97.13206509100314)

    def test_griddata_invert_wterm_noover(self):
        self.actualSetUp(zerow=False)
        gcf, cf = create_awterm_convolutionfunction(
            self.model,
            nw=100,
            wstep=8.0,
            oversampling=1,
            support=32,
            use_aaf=True,
            polarisation_frame=self.vis_pol,
        )

        griddata = create_griddata_from_image(
            self.model, polarisation_frame=self.vis_pol
        )
        griddata, sumwt = grid_visibility_to_griddata(
            self.vis, griddata=griddata, cf=cf
        )
        cim = fft_griddata_to_image(griddata, gcf, gcf)
        cim = normalise_sumwt(cim, sumwt)
        im = convert_polimage_to_stokes(cim)
        if self.persist:
            with tempfile.TemporaryDirectory() as tempdir:
                im.image_acc.export_to_fits(
                    f"{tempdir}/test_gridding_dirty_wterm.fits"
                )
        self.check_peaks(im, 97.1343833)

    def test_griddata_check_cf_grid_wcs(self):
        self.actualSetUp(zerow=False)
        gcf, cf = create_awterm_convolutionfunction(
            self.model,
            nw=11,
            wstep=80.0,
            oversampling=4,
            support=32,
            use_aaf=True,
            polarisation_frame=self.vis_pol,
        )
        griddata = create_griddata_from_image(
            self.model, polarisation_frame=self.vis_pol
        )
        assert (
            cf.convolutionfunction_acc.cf_wcs.wcs.cdelt[0]
            == griddata.griddata_acc.griddata_wcs.wcs.cdelt[0]
        ), str(cf.convolutionfunction_acc.cf_wcs.wcs.cdelt[:2]) + str(
            griddata.griddata_acc.griddata_wcs.wcs.cdelt[:2]
        )
        assert (
            cf.convolutionfunction_acc.cf_wcs.wcs.cdelt[1]
            == griddata.griddata_acc.griddata_wcs.wcs.cdelt[1]
        ), str(cf.convolutionfunction_acc.cf_wcs.wcs.cdelt[:2]) + str(
            griddata.griddata_acc.griddata_wcs.wcs.cdel[:2]
        )

    def test_griddata_predict_aterm(self):
        self.actualSetUp(zerow=True, image_pol=PolarisationFrame("stokesIQUV"))
        make_pb = functools.partial(
            create_pb_generic, diameter=35.0, blockage=0.0, use_local=False
        )
        modelIQUV = convert_stokes_to_polimage(
            self.model, self.vis.visibility_acc.polarisation_frame
        )
        griddata = create_griddata_from_image(
            modelIQUV, polarisation_frame=self.vis_pol
        )
        gcf, cf = create_awterm_convolutionfunction(
            modelIQUV,
            make_pb=make_pb,
            nw=1,
            oversampling=16,
            support=32,
            use_aaf=True,
            polarisation_frame=self.vis_pol,
        )
        griddata = fft_image_to_griddata(modelIQUV, griddata, gcf)
        newvis = degrid_visibility_from_griddata(
            self.vis, griddata=griddata, cf=cf
        )
        qa = newvis.visibility_acc.qa_visibility()
        numpy.testing.assert_allclose(
            qa.data["maxabs"], 1091.515280627418, atol=1e-7, err_msg=f"{qa}"
        )
        numpy.testing.assert_allclose(
            qa.data["minabs"],
            0.00023684744483300332,
            atol=1e-7,
            err_msg=f"{qa}",
        )

    def test_griddata_predict_wterm(self):
        self.actualSetUp(
            zerow=False, image_pol=PolarisationFrame("stokesIQUV")
        )
        gcf, cf = create_awterm_convolutionfunction(
            self.model,
            nw=11,
            wstep=80.0,
            oversampling=4,
            support=32,
            use_aaf=True,
            polarisation_frame=self.vis_pol,
        )
        modelIQUV = convert_stokes_to_polimage(
            self.model, self.vis.visibility_acc.polarisation_frame
        )
        griddata = create_griddata_from_image(
            modelIQUV, polarisation_frame=self.vis_pol
        )
        griddata = fft_image_to_griddata(modelIQUV, griddata, gcf)
        newvis = degrid_visibility_from_griddata(
            self.vis, griddata=griddata, cf=cf
        )
        newvis["vis"].data[...] -= self.vis["vis"].data[...]
        self.plot_vis(newvis, "wterm")
        qa = newvis.visibility_acc.qa_visibility()
        numpy.testing.assert_allclose(
            qa.data["maxabs"], 224.28478109440636, atol=1e-7, err_msg=f"{qa}"
        )
        numpy.testing.assert_allclose(
            qa.data["minabs"], 0.012386229250739898, atol=1e-7, err_msg=f"{qa}"
        )

    def test_griddata_predict_awterm(self):
        self.actualSetUp(
            zerow=False, image_pol=PolarisationFrame("stokesIQUV")
        )
        modelIQUV = convert_stokes_to_polimage(
            self.model, self.vis.visibility_acc.polarisation_frame
        )
        make_pb = functools.partial(
            create_pb_generic, diameter=35.0, blockage=0.0, use_local=False
        )
        pb = make_pb(modelIQUV)
        if self.persist:
            with tempfile.TemporaryDirectory() as tempdir:
                pb.image_acc.export_to_fits(
                    f"{tempdir}/test_gridding_awterm_pb.fits"
                )
        gcf, cf = create_awterm_convolutionfunction(
            self.model,
            make_pb=make_pb,
            nw=100,
            wstep=8.0,
            oversampling=4,
            support=32,
            use_aaf=True,
            polarisation_frame=self.vis_pol,
        )
        griddata = create_griddata_from_image(
            modelIQUV, polarisation_frame=self.vis_pol
        )
        griddata = fft_image_to_griddata(modelIQUV, griddata, gcf)
        newvis = degrid_visibility_from_griddata(
            self.vis, griddata=griddata, cf=cf
        )
        qa = newvis.visibility_acc.qa_visibility()
        numpy.testing.assert_allclose(
            qa.data["maxabs"], 1086.4705273529883, atol=1e-7, err_msg=f"{qa}"
        )
        numpy.testing.assert_allclose(
            qa.data["minabs"], 0.05699706072350753, atol=1e-7, err_msg=f"{qa}"
        )
        self.plot_vis(newvis, "awterm")

    def test_griddata_visibility_weight(self):
        self.actualSetUp(zerow=True, image_pol=PolarisationFrame("stokesIQUV"))
        gcf, cf = create_pswf_convolutionfunction(
            self.model, polarisation_frame=self.vis_pol
        )
        gd = create_griddata_from_image(
            self.model, polarisation_frame=self.vis_pol
        )
        gd_list = [
            grid_visibility_weight_to_griddata(self.vis, gd) for i in range(10)
        ]
        assert numpy.max(numpy.abs(gd_list[0][0]["pixels"].data)) > 10.0
        gd, sumwt = griddata_merge_weights(gd_list)
        self.vis = griddata_visibility_reweight(self.vis, gd)
        gd, sumwt = grid_visibility_to_griddata(self.vis, griddata=gd, cf=cf)
        cim = fft_griddata_to_image(gd, gcf, gcf)
        cim = normalise_sumwt(cim, sumwt)
        im = convert_polimage_to_stokes(cim)
        if self.persist:
            with tempfile.TemporaryDirectory() as tempdir:
                im.image_acc.export_to_fits(
                    f"{tempdir}/test_gridding_dirty_2d_uniform_block.fits"
                )
        self.check_peaks(im, 99.40822097)

    def test_griddata_visibility_weight_I(self):
        self.actualSetUp(zerow=True, image_pol=PolarisationFrame("stokesI"))
        gcf, cf = create_pswf_convolutionfunction(
            self.model, polarisation_frame=self.vis_pol
        )
        gd = create_griddata_from_image(
            self.model, polarisation_frame=self.vis_pol
        )
        gd_list = [
            grid_visibility_weight_to_griddata(self.vis, gd) for i in range(10)
        ]
        assert numpy.max(numpy.abs(gd_list[0][0]["pixels"].data)) > 10.0
        gd, sumwt = griddata_merge_weights(gd_list)
        self.vis = griddata_visibility_reweight(self.vis, gd)
        gd, sumwt = grid_visibility_to_griddata(self.vis, griddata=gd, cf=cf)
        cim = fft_griddata_to_image(gd, gcf, gcf)
        cim = normalise_sumwt(cim, sumwt)
        im = convert_polimage_to_stokes(cim)
        if self.persist:
            with tempfile.TemporaryDirectory() as tempdir:
                im.image_acc.export_to_fits(
                    f"{tempdir}/test_gridding_dirty_2d_IQ_uniform_block.fits"
                )
        self.check_peaks(im, 99.40822097)

    def test_griddata_visibility_weight_IQ(self):
        self.actualSetUp(zerow=True, image_pol=PolarisationFrame("stokesIQ"))
        gcf, cf = create_pswf_convolutionfunction(
            self.model, polarisation_frame=self.vis_pol
        )
        gd = create_griddata_from_image(
            self.model, polarisation_frame=self.vis_pol
        )
        gd_list = [
            grid_visibility_weight_to_griddata(self.vis, gd) for i in range(10)
        ]
        assert numpy.max(numpy.abs(gd_list[0][0]["pixels"].data)) > 10.0
        gd, sumwt = griddata_merge_weights(gd_list)
        self.vis = griddata_visibility_reweight(self.vis, gd)
        gd, sumwt = grid_visibility_to_griddata(self.vis, griddata=gd, cf=cf)
        cim = fft_griddata_to_image(gd, gcf, gcf)
        cim = normalise_sumwt(cim, sumwt)
        im = convert_polimage_to_stokes(cim)
        if self.persist:
            with tempfile.TemporaryDirectory() as tempdir:
                im.image_acc.export_to_fits(
                    f"{tempdir}/test_gridding_dirty_2d_IQ_uniform_block.fits"
                )
        self.check_peaks(im, 99.40822097)

    def test_grid_visibility_weight_to_griddata_ignore_visibilities(self):
        self.actualSetUp(
            zerow=True,
            image_pol=PolarisationFrame("stokesIQUV"),
            test_ignored_visibilities=True,
        )
        grid_data = create_griddata_from_image(
            self.model, polarisation_frame=self.vis_pol
        )
        grid_data_list = [
            grid_visibility_weight_to_griddata(self.vis, grid_data)
            for i in range(10)
        ]
        griddata, sumwt = griddata_merge_weights(grid_data_list)
        # Using sum to judge the correctness after ignored some visbilities
        assert numpy.isclose(
            numpy.sum(numpy.abs(griddata["pixels"].data)),
            numpy.sum(sumwt),
            atol=1e-11,
        )
        assert numpy.isclose(numpy.sum(sumwt), 3327480.0, atol=1e-11)

    def test_griddata_visibility_reweight_ignore_visibilities(self):
        self.actualSetUp(
            zerow=True,
            image_pol=PolarisationFrame("stokesIQUV"),
            test_ignored_visibilities=True,
        )
        grid_data = create_griddata_from_image(
            self.model, polarisation_frame=self.vis_pol
        )
        grid_data_list = [
            grid_visibility_weight_to_griddata(self.vis, grid_data)
            for i in range(10)
        ]
        grid_data, _ = griddata_merge_weights(grid_data_list)
        self.vis = griddata_visibility_reweight(self.vis, grid_data)
        assert numpy.isclose(
            numpy.sum(self.vis.visibility_acc.flagged_imaging_weight),
            10259.6,
            atol=1e-11,
        )

    def test_grid_visibility_to_griddata_ignore_visibilities(self):
        self.actualSetUp(
            zerow=True,
            image_pol=PolarisationFrame("stokesIQUV"),
            test_ignored_visibilities=True,
        )
        # cf: griddata kernel as ConvolutionFunction
        _, cf = create_pswf_convolutionfunction(
            self.model, polarisation_frame=self.vis_pol
        )
        grid_data = create_griddata_from_image(
            self.model, polarisation_frame=self.vis_pol
        )
        grid_data_list = [
            grid_visibility_weight_to_griddata(self.vis, grid_data)
            for i in range(10)
        ]
        grid_data, _ = griddata_merge_weights(grid_data_list)
        self.vis = griddata_visibility_reweight(self.vis, grid_data)
        grid_data, sumwt = grid_visibility_to_griddata(
            self.vis, griddata=grid_data, cf=cf
        )
        # Using sum to judge the correctness after ignored some visbilities
        assert numpy.isclose(
            numpy.sum(numpy.abs(grid_data["pixels"].data)),
            1235383.2942437963,
            atol=1e-11,
        )
        assert numpy.isclose(numpy.sum(sumwt), 10253.599999999533, atol=1e-11)

    def plot_vis(self, newvis, title=""):
        if self.doplot:
            import matplotlib.pyplot as plt

            for pol in range(4):
                plt.plot(newvis.w, numpy.real(newvis.vis[:, pol]), ".")
            plt.xlim(150, 300)
            plt.title("Prediction error for %s gridding" % title)
            plt.xlabel("W (wavelengths)")
            plt.ylabel("Real part of visibility prediction error")
            plt.show(block=False)


if __name__ == "__main__":
    unittest.main()
