"""
Functions that aid fourier transform processing. These are built on top of the core
functions in processing_components.fourier_transforms.

The measurement equation for a sufficently narrow field of view interferometer is:

.. math::

    V(u,v,w) =\\int I(l,m) e^{-2 \\pi j (ul+vm)} dl dm


The measurement equation for a wide field of view interferometer is:

.. math::

    V(u,v,w) =\\int \\frac{I(l,m)}{\\sqrt{1-l^2-m^2}} e^{-2 \\pi j (ul+vm + w(\\sqrt{1-l^2-m^2}-1))} dl dm

This and related modules contain various approachs for dealing with the wide-field problem where the
extra phase term in the Fourier transform cannot be ignored.
"""

__all__ = [
    "shift_vis_to_image",
    "normalise_sumwt",
    "predict_awprojection",
    "invert_awprojection",
    "create_image_from_visibility",
    "advise_wide_field",
    "visibility_recentre",
    "fill_vis_for_psf",
]

import logging

import astropy.units as units
import astropy.wcs as wcs
import numpy
from astropy.wcs.utils import pixel_to_skycoord
from ska_sdp_datamodels.image.image_model import Image
from ska_sdp_datamodels.science_data_model.polarisation_model import (
    PolarisationFrame,
)
from ska_sdp_datamodels.visibility.vis_model import Visibility
from ska_sdp_datamodels import physical_constants
from ska_sdp_datamodels.image.image_create import create_image
from ska_sdp_datamodels.gridded_visibility.grid_vis_create import (
    create_griddata_from_image,
)

from ska_sdp_func_python.griddata.gridding import (
    grid_visibility_to_griddata,
    fft_griddata_to_image,
    degrid_visibility_from_griddata,
    fft_image_to_griddata,
)
from ska_sdp_func_python.visibility.base import phaserotate_visibility
from ska_sdp_func_python.image.operations import (
    convert_polimage_to_stokes,
    convert_stokes_to_polimage,
)

log = logging.getLogger("func-python-logger")


def shift_vis_to_image(
    vis: Visibility, im: Image, tangent: bool = True, inverse: bool = False
) -> Visibility:
    """Shift visibility in place to the phase centre of the Image

    :param vis: Visibility
    :param im: Image model used to determine phase centre
    :param tangent: Is the shift purely on the tangent plane True|False
    :param inverse: Do the inverse operation True|False
    :return: visibility with phase shift applied and phasecentre updated

    """
    # assert isinstance(vis, Visibility), "vis is not a Visibility: %r" % vis

    nchan, npol, ny, nx = im["pixels"].data.shape

    # Convert the FFT definition of the phase center to world coordinates (1 relative)
    # This is the only place in RASCIL where the relationship between the image and visibility
    # frames is defined.

    image_phasecentre = pixel_to_skycoord(
        nx // 2 + 1, ny // 2 + 1, im.image_acc.wcs, origin=1
    )
    if vis.phasecentre.separation(image_phasecentre).rad > 1e-15:
        if inverse:
            log.debug(
                "shift_vis_from_image: shifting phasecentre from image phase centre %s to visibility phasecentre "
                "%s" % (image_phasecentre, vis.phasecentre)
            )
        else:
            log.debug(
                "shift_vis_from_image: shifting phasecentre from vis phasecentre %s to image phasecentre %s"
                % (vis.phasecentre, image_phasecentre)
            )
        vis = phaserotate_visibility(
            vis, image_phasecentre, tangent=tangent, inverse=inverse
        )
        vis.attrs["phasecentre"] = im.image_acc.phasecentre

    return vis


def normalise_sumwt(im: Image, sumwt, min_weight=0.1, flat_sky=False) -> Image:
    """normalise out the sum of weights

    The gridding weights are accumulated as a function of channel and polarisation. This function
    corrects for this sum of weights. The sum of weights can be a 2D array or an image the same
    shape as the image (as for primary beam correction)

    The parameter flat_sky controls whether the sensitivity (sumwt) is divided out pixel by pixel
    or instead the maximum value is divided out.

    :param im: Image, im["pixels"].data has shape [nchan, npol, ny, nx]
    :param sumwt: Sum of weights [nchan, npol] or [nchan, npol, ny, nx]
    :param minwt: Minimum (fractional) weight to be used in dividing by the sumwt images
    :param flat_sky: Make the sky flat? Or the noise flat?
    """
    nchan, npol, _, _ = im["pixels"].data.shape
    assert sumwt is not None
    if isinstance(sumwt, numpy.ndarray):
        # This is the usual case where the primary beams are not included
        assert nchan == sumwt.shape[0]
        assert npol == sumwt.shape[1]
        for chan in range(nchan):
            for pol in range(npol):
                if sumwt[chan, pol] > 0.0:
                    im["pixels"].data[chan, pol, :, :] = (
                        im["pixels"].data[chan, pol, :, :] / sumwt[chan, pol]
                    )
                else:
                    im["pixels"].data[chan, pol, :, :] = 0.0
    elif im["pixels"].data.shape == sumwt["pixels"].data.shape:
        maxwt = numpy.max(sumwt["pixels"].data)
        minwt = min_weight * maxwt
        nchan, npol, ny, nx = sumwt["pixels"].data.shape
        cx = nx // 2
        cy = ny // 2
        for chan in range(nchan):
            for pol in range(npol):
                if flat_sky:
                    norm = numpy.sqrt(
                        sumwt["pixels"].data[chan, pol, cy, cx]
                        * sumwt["pixels"].data[chan, pol, :, :]
                    )
                    im["pixels"].data[chan, pol, :, :][norm > minwt] /= norm[
                        norm > minwt
                    ]
                    im["pixels"].data[chan, pol, :, :][norm <= minwt] /= maxwt
                else:
                    im["pixels"].data[chan, pol, :, :] /= maxwt
                    sumwt["pixels"].data[chan, pol, :, :] /= maxwt
                    sumwt["pixels"].data = numpy.sqrt(sumwt["pixels"].data)
    else:
        raise ValueError(
            "sumwt is not a 2D or 4D array - cannot perform normalisation"
        )

    return im


def predict_awprojection(
    vis: Visibility, model: Image, gcfcf=None, **kwargs
) -> Visibility:
    """Predict using convolutional degridding and an AW kernel

    Note that the gridding correction function (gcf) and convolution function (cf) can be passed
    as a partial function. So the caller must supply a partial function to
    calculate the gcf, cf tuple for an image model.

    :param vis: visibility to be predicted
    :param model: model image
    :return: resulting visibility (in place works)
    """

    if model is None:
        return vis

    assert not numpy.isnan(
        numpy.sum(model["pixels"].data)
    ), "NaNs present in input model"

    _, _, ny, nx = model["pixels"].data.shape

    if gcfcf is None:
        raise ValueError("predict_awprojection: gcfcf not specified")

    gcf, cf = gcfcf(model)

    griddata = create_griddata_from_image(
        model, polarisation_frame=vis.visibility_acc.polarisation_frame
    )
    polmodel = convert_stokes_to_polimage(
        model, vis.visibility_acc.polarisation_frame
    )
    griddata = fft_image_to_griddata(polmodel, griddata, gcf)
    vis = degrid_visibility_from_griddata(vis, griddata=griddata, cf=cf)

    # Now we can shift the visibility from the image frame to the original visibility frame
    svis = shift_vis_to_image(vis, model, tangent=True, inverse=True)

    return svis


def invert_awprojection(
    vis: Visibility,
    im: Image,
    dopsf: bool = False,
    normalise: bool = True,
    gcfcf=None,
    **kwargs
) -> (Image, numpy.ndarray):
    """Invert using convolutional degridding and an AW kernel

    Use the image im as a template. Do PSF in a separate call.

    Note that the gridding correction function (gcf) and convolution function (cf) can be passed
    as a partial function. So the caller must supply a partial function to
    calculate the gcf, cf tuple for an image model.

    :param vis: visibility to be inverted
    :param im: image template (not changed)
    :param dopsf: Make the psf instead of the dirty image
    :param normalise: normalise by the sum of weights (True)
    :param gcfcf: (Grid correction function i.e. in image space, Convolution function i.e. in uv space)
    :return: resulting image

    """

    svis = vis.copy(deep=True)

    if dopsf:
        svis = fill_vis_for_psf(svis)

    svis = shift_vis_to_image(svis, im, tangent=True, inverse=False)

    griddata = create_griddata_from_image(
        im, polarisation_frame=vis.visibility_acc.polarisation_frame
    )
    if gcfcf is None:
        raise ValueError("invert_awprojection: gcfcf not specified")

    gcf, cf = gcfcf(im)
    griddata, sumwt = grid_visibility_to_griddata(
        svis, griddata=griddata, cf=cf
    )
    result = fft_griddata_to_image(griddata, im, gcf)

    if normalise:
        result = normalise_sumwt(result, sumwt)

    result = convert_polimage_to_stokes(result, **kwargs)

    assert not numpy.isnan(
        numpy.sum(result["pixels"].data)
    ), "NaNs present in output image"

    return result, sumwt


def fill_vis_for_psf(svis):
    """Fill the visibility for calculation of PSF

    :param im:
    :param svis:
    :return: visibility with unit vis
    """
    if svis.visibility_acc.polarisation_frame == PolarisationFrame("linear"):
        svis["vis"].data[..., 0] = 1.0 + 0.0j
        svis["vis"].data[..., 1:3] = 0.0 + 0.0j
        svis["vis"].data[..., 3] = 1.0 + 0.0j
    elif svis.visibility_acc.polarisation_frame == PolarisationFrame(
        "circular"
    ):
        svis["vis"].data[..., 0] = 1.0 + 0.0j
        svis["vis"].data[..., 1:3] = 0.0 + 0.0j
        svis["vis"].data[..., 3] = 1.0 + 0.0j
    elif svis.visibility_acc.polarisation_frame == PolarisationFrame(
        "linearnp"
    ):
        svis["vis"].data[...] = 1.0 + 0.0j
    elif svis.visibility_acc.polarisation_frame == PolarisationFrame(
        "circularnp"
    ):
        svis["vis"].data[...] = 1.0 + 0.0j
    elif svis.visibility_acc.polarisation_frame == PolarisationFrame(
        "stokesI"
    ):
        svis["vis"].data[...] = 1.0 + 0.0j
    else:
        raise ValueError(
            "Cannot calculate PSF for {}".format(
                svis.visibility_acc.polarisation_frame
            )
        )

    return svis


def create_image_from_visibility(vis: Visibility, **kwargs) -> Image:
    """Make an empty image from params and Visibility

    This makes an empty, template image consistent with the visibility, allowing optional overriding of select
    parameters. This is a convenience function and does not transform the visibilities.

    :param vis:
    :param phasecentre: Phasecentre (Skycoord)
    :param channel_bandwidth: Channel width (Hz)
    :param cellsize: Cellsize (radians)
    :param npixel: Number of pixels on each axis (512)
    :param frame: Coordinate frame for WCS (ICRS)
    :param equinox: Equinox for WCS (2000.0)
    :param nchan: Number of image channels (Default is 1 -> MFS)
    :return: image

    See also
        :py:func:`rascil.processing_components.image.operations.create_image`
    """
    log.debug(
        "create_image_from_visibility: Parsing parameters to get definition of WCS"
    )

    try:
        image_centre = kwargs["imagecentre"]
    except KeyError:
        log.info(
            "create_image_from_visibility: no image_centre given, setting default value"
        )
        image_centre = vis.phasecentre
    try:
        phase_centre = kwargs["phasecentre"]
    except KeyError:
        log.info(
            "create_image_from_visibility: no phase_centre given, setting default value"
        )
        phase_centre = vis.phasecentre

    # Spectral processing options
    ufrequency = numpy.unique(vis["frequency"].data)
    try:
        frequency = kwargs["frequency"]
    except KeyError:
        log.info(
            "create_image_from_visibility: no frequency given, setting default value"
        )
        frequency = vis["frequency"].data

    vnchan = len(ufrequency)

    try:
        inchan = kwargs["nchan"]
    except KeyError:
        log.info(
            "create_image_from_visibility: no inchan given, setting default value"
        )
        inchan = vnchan

    reffrequency = frequency[0] * units.Hz

    try:
        channel_bandwidth = kwargs["channel_bandwidth"]
    except KeyError:
        log.info(
            "create_image_from_visibility: no channel_bandwidth given, setting default value"
        )
        channel_bandwidth = vis["channel_bandwidth"].data.flat[0]

    channel_bandwidth = channel_bandwidth * units.Hz

    if (inchan == vnchan) and vnchan > 1:
        log.debug(
            "create_image_from_visibility: Defining %d channel Image at %s, starting frequency %s, and bandwidth %s"
            % (inchan, image_centre, reffrequency, channel_bandwidth)
        )
    elif (inchan == 1) and vnchan > 1:
        assert (
            numpy.abs(channel_bandwidth) > 0.0
        ), "Channel width must be non-zero for mfs mode"
        log.debug(
            "create_image_from_visibility: Defining single channel MFS Image at %s, starting frequency %s, "
            "and bandwidth %s"
            % (image_centre, reffrequency, channel_bandwidth)
        )
    elif inchan > 1 and vnchan > 1:
        assert (
            numpy.abs(channel_bandwidth) > 0.0
        ), "Channel width must be non-zero for mfs mode"
        log.debug(
            "create_image_from_visibility: Defining multi-channel MFS Image at %s, starting frequency %s, "
            "and bandwidth %s"
            % (image_centre, reffrequency, channel_bandwidth)
        )
    elif (inchan == 1) and (vnchan == 1):
        assert (
            numpy.abs(channel_bandwidth) > 0.0
        ), "Channel width must be non-zero for mfs mode"
        log.debug(
            "create_image_from_visibility: Defining single channel Image at %s, starting frequency %s, "
            "and bandwidth %s"
            % (image_centre, reffrequency, channel_bandwidth)
        )
    else:
        raise ValueError(
            "create_image_from_visibility: unknown spectral mode inchan = {}, vnchan = {} ".format(
                inchan, vnchan
            )
        )

    # Image sampling options
    try:
        npixel = kwargs["npixel"]
    except KeyError:
        log.info(
            "create_image_from_visibility: no npixel given, setting default value"
        )
        npixel = 512

    uvmax = numpy.max((numpy.abs(vis.visibility_acc.uvw_lambda[..., 0:2])))
    log.debug("create_image_from_visibility: uvmax = %f wavelengths" % uvmax)
    criticalcellsize = 1.0 / (uvmax * 2.0)
    log.debug(
        "create_image_from_visibility: Critical cellsize = %f radians, %f degrees"
        % (criticalcellsize, criticalcellsize * 180.0 / numpy.pi)
    )

    try:
        cellsize = kwargs["cellsize"]
    except KeyError:
        log.info(
            "create_image_from_visibility: no cellsize given, setting default value"
        )
        cellsize = 0.5 * criticalcellsize

    log.debug(
        "create_image_from_visibility: Cellsize          = %g radians, %g degrees"
        % (cellsize, cellsize * 180.0 / numpy.pi)
    )
    try:
        override_cellsize = kwargs["override_cellsize"]
    except KeyError:
        log.info(
            "create_image_from_visibility: no override_cellsize given, setting default value"
        )
        override_cellsize = True

    if (override_cellsize and cellsize > criticalcellsize) or (
        cellsize == 0.0
    ):
        log.debug(
            "create_image_from_visibility: Resetting cellsize %g radians to criticalcellsize %g radians"
            % (cellsize, criticalcellsize)
        )
        cellsize = criticalcellsize
    try:
        pol_frame = kwargs["polarisation_frame"]
    except KeyError:
        log.info(
            "create_image_from_visibility: no pol_frame given, setting default value"
        )
        pol_frame = PolarisationFrame("stokesI")

    inpol = pol_frame.npol

    # Now we can define the WCS, which is a convenient place to hold the info above
    # Beware of python indexing order! wcs and the array have opposite ordering
    shape = [inchan, inpol, npixel, npixel]
    log.debug("create_image_from_visibility: image shape is %s" % str(shape))
    w = wcs.WCS(naxis=4)
    # The negation in the longitude is needed by definition of RA, DEC
    w.wcs.cdelt = [
        -cellsize * 180.0 / numpy.pi,
        cellsize * 180.0 / numpy.pi,
        1.0,
        channel_bandwidth.to(units.Hz).value,
    ]
    # The numpy definition of the phase centre of an FFT is n // 2 (0 - rel) so that's what we use for
    # the reference pixel. We have to use 0 rel everywhere.
    w.wcs.crpix = [npixel // 2 + 1, npixel // 2 + 1, 1.0, 1.0]
    w.wcs.ctype = ["RA---SIN", "DEC--SIN", "STOKES", "FREQ"]
    w.wcs.crval = [
        phase_centre.ra.deg,
        phase_centre.dec.deg,
        1.0,
        reffrequency.to(units.Hz).value,
    ]
    w.naxis = 4

    try:
        w.wcs.radesys = kwargs["frame"]
    except KeyError:
        log.info(
            "create_image_from_visibility: no radesys given, setting default value"
        )
        w.wcs.radesys = "ICRS"

    try:
        w.wcs.equinox = kwargs["equinox"]
    except KeyError:
        log.info(
            "create_image_from_visibility: no equinox given, setting default value"
        )
        w.wcs.equinox = 2000.0

    try:
        chunksize = kwargs["chunksize"]
    except KeyError:
        log.info(
            "create_image_from_visibility: no chunksize given, setting default value"
        )
        chunksize = None
    im = create_image(npixel, cellsize, phase_centre)
    return im


def advise_wide_field(
    vis: Visibility,
    delA=0.02,
    oversampling_synthesised_beam=3.0,
    guard_band_image=6.0,
    facets=1,
    wprojection_planes=1,
    verbose=True,
):
    """Advise on parameters for wide field imaging.

    Calculate sampling requirements on various parameters

    For example::

        advice = advise_wide_field(vis, delA)
        try:
            wstep = kwargs["wstep"]
        except KeyError:
            wstep = advice['w_sampling_primary_beam'])


    :param vis:
    :param delA: Allowed coherence loss (def: 0.02)
    :param oversampling_synthesised_beam: Oversampling of the synthesized beam (def: 3.0)
           If the value <=2, some visibilities in gridding would be discarded.
    :param guard_band_image: Number of primary beam half-widths-to-half-maximum to image (def: 6)
    :param facets: Number of facets on each axis
    :param wprojection_planes: Number of planes in wprojection
    :return: dict of advice
    """

    isblock = isinstance(vis, Visibility)

    max_wavelength = physical_constants.c_m_s / numpy.min(vis.frequency.data)
    if verbose:
        log.info(
            "advise_wide_field: (max_wavelength) Maximum wavelength %.3f (meters)"
            % (max_wavelength)
        )

    min_wavelength = physical_constants.c_m_s / numpy.max(vis.frequency.data)
    if verbose:
        log.info(
            "advise_wide_field: (min_wavelength) Minimum wavelength %.3f (meters)"
            % (min_wavelength)
        )

    maximum_baseline = (
        numpy.max(numpy.abs(vis["uvw"].data)) / min_wavelength
    )  # Wavelengths
    maximum_w = (
        numpy.max(numpy.abs(vis.visibility_acc.w.data)) / min_wavelength
    )  # Wavelengths

    if verbose:
        log.info(
            "advise_wide_field: (maximum_baseline) Maximum baseline %.1f (wavelengths)"
            % (maximum_baseline)
        )
    assert maximum_baseline > 0.0, "Error in UVW coordinates: all uvw are zero"

    if verbose:
        log.info(
            "advise_wide_field: (maximum_w) Maximum w %.1f (wavelengths)"
            % (maximum_w)
        )

    diameter = numpy.min(vis.attrs["configuration"].diameter.data)
    if verbose:
        log.info(
            "advise_wide_field: (diameter) Station/dish diameter %.1f (meters)"
            % (diameter)
        )
    assert diameter > 0.0, "Station/dish diameter must be greater than zero"

    primary_beam_fov = max_wavelength / diameter
    if verbose:
        log.info(
            "advise_wide_field: (primary_beam_fov) Primary beam %s"
            % (rad_deg_arcsec(primary_beam_fov))
        )

    image_fov = primary_beam_fov * guard_band_image
    if verbose:
        log.info(
            "advise_wide_field: (image_fov) Image field of view %s"
            % (rad_deg_arcsec(image_fov))
        )

    facet_fov = primary_beam_fov * guard_band_image / facets
    if facets > 1:
        if verbose:
            log.info(
                "advise_wide_field: (facet_fov) Facet field of view %s"
                % (rad_deg_arcsec(facet_fov))
            )

    synthesized_beam = 1.0 / (maximum_baseline)
    if verbose:
        log.info(
            "advise_wide_field: (synthesized_beam) Synthesized beam %s"
            % (rad_deg_arcsec(synthesized_beam))
        )

    cellsize = synthesized_beam / oversampling_synthesised_beam
    if verbose:
        log.info(
            "advise_wide_field: (cellsize) Cellsize %s"
            % (rad_deg_arcsec(cellsize))
        )
        log.info("")

    def pwr2(n):
        ex = numpy.ceil(numpy.log(n) / numpy.log(2.0)).astype("int")
        best = numpy.power(2, ex)
        return best

    def pwr23(n):
        ex = numpy.ceil(numpy.log(n) / numpy.log(2.0)).astype("int")
        best = numpy.power(2, ex)
        if best * 3 // 4 >= n:
            best = best * 3 // 4
        return best

    def pwr2345(n):
        # If pyfftw has been installed, next_fast_len would return the len of best performance
        try:
            import pyfftw

            best = pyfftw.next_fast_len(n)
        except ImportError:
            pyfftw = None
            number = numpy.array([2, 3, 4, 5])
            ex = numpy.ceil(numpy.log(n) / numpy.log(number)).astype("int")
            best = min(numpy.power(number[:], ex[:]))
        return best

    npixels = int(round(image_fov / cellsize))
    if verbose:
        log.info(
            "advice_wide_field: (npixels) Npixels per side = %d" % (npixels)
        )

    npixels2 = pwr2(npixels)
    if verbose:
        log.info(
            "advice_wide_field: (npixels2) Npixels (power of 2) per side = %d"
            % (npixels2)
        )

    npixels23 = pwr23(npixels)
    if verbose:
        log.info(
            "advice_wide_field: (npixels23) Npixels (power of 2, 3) per side = %d"
            % (npixels23)
        )

    npixels_min = pwr2345(npixels)
    if verbose:
        log.info(
            "advice_wide_field: (npixels_min) Npixels (power of 2, 3, 4, 5) per side = %d"
            % (npixels_min)
        )

    # Following equation is from Cornwell, Humphreys, and Voronkov (2012) (equation 24)
    # We will assume that the constraint holds at one quarter the entire FOV i.e. that
    # the full field of view includes the entire primary beam

    w_sampling_image = numpy.sqrt(2.0 * delA) / (numpy.pi * image_fov**2)
    if verbose:
        log.info(
            "\nadvice_wide_field: (w_sampling_image) W sampling for full image = %.1f (wavelengths)"
            % (w_sampling_image)
        )

    if facets > 1:
        w_sampling_facet = numpy.sqrt(2.0 * delA) / (numpy.pi * facet_fov**2)
        if verbose:
            log.info(
                "advice_wide_field: (w_sampling_facet) W sampling for facet = %.1f (wavelengths)"
                % (w_sampling_facet)
            )
    else:
        w_sampling_facet = w_sampling_image

    w_sampling_primary_beam = numpy.sqrt(2.0 * delA) / (
        numpy.pi * primary_beam_fov**2
    )
    if verbose:
        log.info(
            "advice_wide_field: (w_sampling_primary_beam) W sampling for primary beam = %.1f (wavelengths)"
            % (w_sampling_primary_beam)
        )

    time_sampling_image = 86400.0 * (synthesized_beam / image_fov)
    if verbose:
        log.info(
            "advice_wide_field: (time_sampling_image) Time sampling for full image = %.1f (s)"
            % (time_sampling_image)
        )

    if facets > 1:
        time_sampling_facet = 86400.0 * (synthesized_beam / facet_fov)
        if verbose:
            log.info(
                "advice_wide_field: (time_sampling_facet) Time sampling for facet = %.1f (s)"
                % (time_sampling_facet)
            )

    time_sampling_primary_beam = 86400.0 * (
        synthesized_beam / primary_beam_fov
    )
    if verbose:
        log.info(
            "advice_wide_field: (time_sampling_primary_beam) Time sampling for primary beam = %.1f (s)"
            % (time_sampling_primary_beam)
        )

    max_freq = numpy.max(vis["frequency"].data)

    freq_sampling_image = max_freq * (synthesized_beam / image_fov)
    if verbose:
        log.info(
            "advice_wide_field: (freq_sampling_image) Frequency sampling for full image = %.1f (Hz)"
            % (freq_sampling_image)
        )

    if facets > 1:
        freq_sampling_facet = max_freq * (synthesized_beam / facet_fov)
        if verbose:
            log.info(
                "advice_wide_field: (freq_sampling_facet) Frequency sampling for facet = %.1f (Hz)"
                % (freq_sampling_facet)
            )

    freq_sampling_primary_beam = max_freq * (
        synthesized_beam / primary_beam_fov
    )
    if verbose:
        log.info(
            "advice_wide_field: (freq_sampling_primary_beam) Frequency sampling for primary beam = %.1f (Hz)"
            % (freq_sampling_primary_beam)
        )
        log.info("")

    wstep_primary_beam = w_sampling_primary_beam
    vis_slices_primary_beam = max(1, int(2 * maximum_w / wstep_primary_beam))
    wprojection_planes_primary_beam = vis_slices_primary_beam
    nwpixels_primary_beam = int(
        2.0 * wprojection_planes_primary_beam * primary_beam_fov
    )
    nwpixels_primary_beam = nwpixels_primary_beam - nwpixels_primary_beam % 2
    if verbose:
        log.info(
            "advice_wide_field: (vis_slices_primary_beam) Number of planes in w stack %d (primary beam)"
            % (vis_slices_primary_beam)
        )
        log.info(
            "advice_wide_field: (wprojection_planes_primary_beam) Number of planes in w projection %d (primary beam)"
            % (wprojection_planes_primary_beam)
        )
        log.info(
            "advice_wide_field: (nwpixels_primary_beam) W support = %d (pixels) (primary beam)"
            % nwpixels_primary_beam
        )
        log.info("")

    wstep_image = w_sampling_image
    vis_slices_image = max(1, int(2 * maximum_w / wstep_image))
    wprojection_planes_image = vis_slices_image
    nwpixels_image = int(2.0 * wprojection_planes_image * image_fov)
    nwpixels_image = nwpixels_image - nwpixels_image % 2
    if verbose:
        log.info(
            "advice_wide_field: (vis_slices_image) Number of planes in w stack %d (primary beam)"
            % (vis_slices_image)
        )
        log.info(
            "advice_wide_field: (wprojection_planes_image) Number of planes in w projection %d (image)"
            % (wprojection_planes_image)
        )
        log.info(
            "advice_wide_field: (nwpixels_image) W support = %d (pixels) (image)"
            % nwpixels_image
        )
        log.info("")
        log.info(
            "advise_wide_field: by default, using primary beam to advise on w sampling parameters"
        )

    wstep = wstep_primary_beam
    vis_slices = vis_slices_primary_beam
    wprojection_planes = wprojection_planes_primary_beam
    nwpixels = nwpixels_primary_beam

    result = locals()

    keys = [
        "delA",
        "oversampling_synthesised_beam",
        "guard_band_image",
        "facets",
        "wprojection_planes",
        "verbose",
        "max_wavelength",
        "min_wavelength",
        "maximum_baseline",
        "maximum_w",
        "diameter",
        "primary_beam_fov",
        "image_fov",
        "facet_fov",
        "synthesized_beam",
        "cellsize",
        "npixels",
        "npixels2",
        "npixels23",
        "npixels_min",
        "w_sampling_image",
        "w_sampling_facet",
        "w_sampling_primary_beam",
        "time_sampling_image",
        "time_sampling_primary_beam",
        "max_freq",
        "freq_sampling_image",
        "freq_sampling_primary_beam",
        "wstep_primary_beam",
        "vis_slices_primary_beam",
        "wprojection_planes_primary_beam",
        "nwpixels_primary_beam",
        "wstep_image",
        "vis_slices_image",
        "wprojection_planes_image",
        "nwpixels_image",
        "wstep",
        "vis_slices",
        "nwpixels",
    ]

    return {your_key: result[your_key] for your_key in keys}


def rad_deg_arcsec(x):
    """Stringify x in radian and degress forms"""
    return "%.3g (rad) %.3g (deg) %.3g (asec)" % (
        x,
        180.0 * x / numpy.pi,
        3600.0 * 180.0 * x / numpy.pi,
    )


def visibility_recentre(uvw, dl, dm):
    """Compensate for kernel re-centering - see `w_kernel_function`.

    :param uvw: Visibility coordinates
    :param dl: Horizontal shift to compensate for
    :param dm: Vertical shift to compensate for
    :returns: Visibility coordinates re-centrered on the peak of their w-kernel
    """

    u, v, w = numpy.hsplit(
        uvw, 3
    )  # pylint: disable=unbalanced-tuple-unpacking
    return numpy.hstack([u - w * dl, v - w * dm, w])
