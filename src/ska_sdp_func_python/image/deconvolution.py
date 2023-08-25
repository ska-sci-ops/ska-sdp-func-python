# pylint: disable=too-many-lines

"""
Image deconvolution functions.
"""

__all__ = [
    "deconvolve_cube",
    "deconvolve_list",
    "fit_psf",
    "restore_list",
    "restore_cube",
]

import logging
import warnings
from typing import List

import numpy
from astropy.convolution import Gaussian2DKernel, convolve_fft
from astropy.modeling import fitting, models
from scipy.optimize import minpack
from ska_sdp_datamodels.image.image_model import Image
from ska_sdp_datamodels.science_data_model.polarisation_model import (
    PolarisationFrame,
)

from ska_sdp_func_python.image.cleaners import (
    hogbom,
    hogbom_complex,
    msclean,
    msmfsclean,
)
from ska_sdp_func_python.image.gather_scatter import (
    image_gather_channels,
    image_scatter_channels,
)
from ska_sdp_func_python.image.operations import (
    convert_clean_beam_to_degrees,
    convert_clean_beam_to_pixels,
)
from ska_sdp_func_python.image.taylor_terms import (
    calculate_image_list_frequency_moments,
    calculate_image_list_from_frequency_taylor_terms,
)

log = logging.getLogger("func-python-logger")


def deconvolve_list(
    dirty_list: List[Image],
    psf_list: List[Image],
    sensitivity_list: List[Image] = None,
    prefix="",
    **kwargs,
) -> (List[Image], List[Image]):
    """
    Clean using a variety of algorithms.
    The algorithms available are:

     hogbom: Hogbom CLEAN See: Hogbom CLEAN A&A Suppl, 15, 417, (1974)

     hogbom-complex: Complex Hogbom CLEAN of stokesIQUV image

     msclean: MultiScale CLEAN See: Cornwell, T.J., Multiscale CLEAN
     (IEEE Journal of Selected Topics in Sig Proc,
     2008 vol. 2 pp. 793-801)

     mfsmsclean, msmfsclean, mmclean: MultiScale Multi-Frequency
     See: U. Rau and T. J. Cornwell,
     “A multi-scale multi-frequency deconvolution algorithm
     for synthesis imaging in radio interferometry,”
     A&A 532, A71 (2011).

    For example::

         comp, residual = deconvolve_list(dirty_list, psf_list, niter=1000,
                                gain=0.7, algorithm='msclean',
                                scales=[0, 3, 10, 30], threshold=0.01)

    For the MFS clean, the psf must have number of channels >= 2 * nmoment.

    :param dirty_list: list of dirty image
    :param psf_list: list of point spread function
    :param sensitivity_list: List of Sensitivity image
                (i.e. inverse noise level)
    :param prefix: Informational message for logging
    :param window_shape: Window description
    :param mask: Window in the form of an image,
                 overrides window_shape
    :param algorithm: Cleaning algorithm:
                'msclean'|'hogbom'|'hogbom-complex'|'mfsmsclean'
    :param gain: loop gain (float) 0.7
    :param threshold: Clean threshold (0.0)
    :param fractional_threshold: Fractional threshold (0.01)
    :param scales: Scales (in pixels) for multiscale ([0, 3, 10, 30])
    :param nmoment: Number of frequency moments (default 3)
    :param findpeak: Method of finding peak in mfsclean:
                    'Algorithm1'|'ASKAPSoft'|'CASA'|'RASCIL',
                    Default is RASCIL.
    :return: component image_list, residual image_list

     See also
        :py:func:`ska_sdp_func_python.image.cleaners.hogbom`
        :py:func:`ska_sdp_func_python.image.cleaners.hogbom_complex`
        :py:func:`ska_sdp_func_python.image.cleaners.msclean`
        :py:func:`ska_sdp_func_python.image.cleaners.msmfsclean`

    """

    window_shape = kwargs.get("window_shape", None)
    window_list = find_window_list(
        dirty_list, prefix, window_shape=window_shape
    )

    check_psf_peak(psf_list)

    psf_support = kwargs.get("psf_support", None)
    psf_list = bound_psf_list(
        dirty_list, prefix, psf_list, psf_support=psf_support
    )

    check_psf_peak(psf_list)

    algorithm = kwargs.get("algorithm", "msclean")

    if algorithm == "msclean":
        comp_image_list, residual_image_list = msclean_kernel_list(
            dirty_list,
            prefix,
            psf_list,
            window_list,
            sensitivity_list,
            **kwargs,
        )
    elif algorithm in ("msmfsclean", "mfsmsclean", "mmclean"):
        comp_image_list, residual_image_list = mmclean_kernel_list(
            dirty_list,
            prefix,
            psf_list,
            window_list,
            sensitivity_list,
            **kwargs,
        )
    elif algorithm == "hogbom":
        comp_image_list, residual_image_list = hogbom_kernel_list(
            dirty_list, prefix, psf_list, window_list, **kwargs
        )
    elif algorithm == "hogbom-complex":
        comp_image_list, residual_image_list = complex_hogbom_kernel_list(
            dirty_list, psf_list, window_list, **kwargs
        )
    else:
        raise ValueError(
            f"deconvolve_cube {prefix}: Unknown algorithm {algorithm}"
        )

    log.info("deconvolve_cube %s: Deconvolution finished", prefix)

    return comp_image_list, residual_image_list


def radler_deconvolve_list(
    dirty_list: List[Image],
    psf_list: List[Image],
    **kwargs,
) -> List[Image]:
    """
    Clean using the Radler module, using various algorithms.

    The algorithms available are
    (see: https://radler.readthedocs.io/en/latest/tree/cpp/algorithms.html):

     msclean
     iuwt
     more_sane
     generic_clean

    For example::

         comp = radler_deconvolve_list(dirty_list, psf_list, niter=1000,
                        gain=0.7, algorithm='msclean',
                        scales=[0, 3, 10, 30], threshold=0.01)

    :param dirty_list: list of dirty image
    :param psf_list: list of point spread function
    :param prefix: Informational message for logging
    :param algorithm: Cleaning algorithm:
                'msclean'|'iuwt'|'more_sane'|'generic_clean'
    :param gain: loop gain (float) 0.7
    :param threshold: Clean threshold (0.0)
    :param scales: Scales (in pixels) for multiscale ([0, 3, 10, 30])
    :param niter: Maximum number of iterations
    :param cellsize: Cell size of each pixel in the image
    :return: component image_list

    """
    import radler as rd  # pylint: disable=import-error,import-outside-toplevel

    algorithm = kwargs.get("algorithm", "msclean")
    n_iterations = kwargs.get("niter", 500)
    clean_threshold = kwargs.get("threshold", 0.001)
    loop_gain = kwargs.get("gain", 0.7)
    ms_scales = kwargs.get("scales", [])
    cellsize = kwargs.get("cellsize", 0.005)

    settings = rd.Settings()
    settings.trimmed_image_width = dirty_list[0].pixels.shape[2]
    settings.trimmed_image_height = dirty_list[0].pixels.shape[3]
    settings.pixel_scale.x = cellsize
    settings.pixel_scale.y = cellsize
    settings.minor_iteration_count = n_iterations
    settings.threshold = clean_threshold
    settings.minor_loop_gain = loop_gain
    if algorithm == "msclean":
        settings.algorithm_type = rd.AlgorithmType.multiscale
        if len(ms_scales) > 0:
            settings.multiscale.scale_list = ms_scales
    elif algorithm == "iuwt":
        settings.algorithm_type = rd.AlgorithmType.iuwt
    elif algorithm == "more_sane":
        settings.algorithm_type = rd.AlgorithmType.more_sane
    elif algorithm == "generic_clean":
        settings.algorithm_type = rd.AlgorithmType.generic_clean
    else:
        raise ValueError(
            f"imaging_deconvolve with radler: Unknown algorithm {algorithm}"
        )

    comp_image_list = []
    for i, dirty in enumerate(dirty_list):
        psf_radler = (
            psf_list[i].pixels.to_numpy().astype(numpy.float32).squeeze()
        )
        dirty_radler = dirty.pixels.to_numpy().astype(numpy.float32).squeeze()
        restored_radler = numpy.zeros_like(dirty_radler)

        radler_object = rd.Radler(
            settings,
            psf_radler,
            dirty_radler,
            restored_radler,
            0.0,
            rd.Polarization.stokes_i,
        )
        reached_threshold = False
        reached_threshold = radler_object.perform(reached_threshold, 0)

        # create an empty image
        # TODO: Need to test this after Radler test is developed

        x_im = Image.constructor(
            data=numpy.zeros_like(dirty["pixels"].data),
            polarisation_frame=dirty.image_acc.polarisation_frame,
            wcs=dirty.image_acc.wcs,
            clean_beam=dirty.attrs["clean_beam"],
        )
        x_im["pixels"].data = numpy.expand_dims(restored_radler, axis=(0, 1))
        comp_image_list.append(x_im)

    return comp_image_list


def check_psf_peak(psf_list):
    """Check that all PSFs in a list have unit peak.

    :param psf_list: List of PSF images
    :return: True if peak exists
    """
    for ipsf, psf in enumerate(psf_list):
        pmax = psf["pixels"].data.max()
        numpy.testing.assert_approx_equal(
            pmax,
            1.0,
            err_msg=f"check_psf_peak: PSF {ipsf} "
            f"does not have unit peak {pmax}",
            significant=6,
        )


def find_window_list(dirty_list, prefix, window_shape=None, **kwargs):
    """Find a clean window from a dirty image.

     The values for window_shape are:
         "quarter" - Inner quarter of image
         "no_edge" - all but window_edge pixels around the perimeter
         mask - If an Image, use as the window (overrides other options)
         None - Entire image

    :param dirty_list: Image of the dirty image
    :param prefix: Informational prefix for log messages
    :param window_shape: Shape of window
    :param kwargs:
    :return: Numpy array
    """
    if window_shape is None:
        log.info("deconvolve_cube %s: Cleaning entire image", prefix)
        return None

    windows = []
    for _, dirty in enumerate(dirty_list):
        if window_shape == "quarter":
            log.info("deconvolve_cube %s: window is inner quarter", prefix)
            qx = dirty["pixels"].shape[3] // 4
            qy = dirty["pixels"].shape[2] // 4
            window_array = numpy.zeros_like(dirty["pixels"].data)
            window_array[..., (qy + 1) : 3 * qy, (qx + 1) : 3 * qx] = 1.0
            log.info(
                "deconvolve_cube %s: Cleaning inner quarter of each sky plane",
                prefix,
            )
        elif window_shape == "no_edge":
            edge = kwargs.get("window_edge", 16)
            nx = dirty["pixels"].shape[3]
            ny = dirty["pixels"].shape[2]
            window_array = numpy.zeros_like(dirty["pixels"].data)
            window_array[
                ..., (edge + 1) : (ny - edge), (edge + 1) : (nx - edge)
            ] = 1.0
            log.info(
                "deconvolve_cube %s: Window omits "
                "%d-pixel edge of each sky plane",
                prefix,
                edge,
            )
        else:
            raise ValueError(f"Window shape {window_shape} is not recognized")

        mask = kwargs.get("mask", None)
        if isinstance(mask, Image):
            if window_array is not None:
                log.warning(
                    "deconvolve_cube %s: Overriding "
                    "window_shape with mask image",
                    prefix,
                )
                window_array = mask["pixels"].data
        if window_array is not None:
            window_image = Image.constructor(
                data=window_array,
                polarisation_frame=dirty.image_acc.polarisation_frame,
                wcs=dirty.image_acc.wcs,
            )
        else:
            window_image = None

        windows.append(window_image)
    return windows


def bound_psf_list(dirty_list, prefix, psf_list, psf_support=None):
    """Calculate the PSF within a given support.

    :param dirty_list: Dirty image, used for default sizes
    :param prefix: Informational prefix to log messages
    :param psf_list: Point Spread Function
    :param psf_support: The half width of a box centered on the psf centre
    :return: psf: bounded point spread function
                  (i.e. with smaller size in x and y)
    """
    psfs = []
    for channel, dirty in enumerate(dirty_list):
        psf = psf_list[channel]
        if psf_support is None:
            psf_support = max(
                dirty["pixels"].shape[2] // 2, dirty["pixels"].shape[3] // 2
            )

        if (psf_support <= psf["pixels"].shape[2] // 2) and (
            (psf_support <= psf["pixels"].shape[3] // 2)
        ):
            centre = [psf["pixels"].shape[2] // 2, psf["pixels"].shape[3] // 2]
            psf = psf.isel(
                x=slice((centre[0] - psf_support), (centre[0] + psf_support)),
                y=slice((centre[1] - psf_support), (centre[1] + psf_support)),
            )
            log.debug(
                "deconvolve_cube %s: PSF support = +/- %d pixels",
                prefix,
                psf_support,
            )
            log.debug(
                "deconvolve_cube %s: PSF shape %s",
                prefix,
                str(psf["pixels"].data.shape),
            )
        else:
            log.info("Using entire psf for deconvolution")
        psfs.append(psf)
    return psfs


def complex_hogbom_kernel_list(
    dirty_list: List[Image],
    psf_list: List[Image],
    window_list: List[Image],
    **kwargs,
):
    """
    Complex Hogbom CLEAN of stokesIQUV image,
    operating of lists of single frequency images.

    :param dirty_list: Image dirty image
    :param psf_list: Image Point Spread Function
    :param window_list: Window array (Bool) - clean where True
    :param gain: loop gain (float) 0.q
    :param threshold: Clean threshold (0.0)
    :param fractional_threshold: Fractional threshold (0.01)
    :return: component image_list, residual image_list
    """

    log.info(
        "complex_hogbom_kernel_list: Starting Hogbom-complex "
        "clean of each channel separately"
    )

    fracthresh, gain, niter, thresh, _ = common_arguments(**kwargs)

    comp_images = []
    residual_images = []

    # Clean each dirty image in the list
    for channel, dirty in enumerate(dirty_list):
        psf = psf_list[channel]
        window = window_list[channel]
        comp_array = numpy.zeros(dirty["pixels"].data.shape)
        residual_array = numpy.zeros(dirty["pixels"].data.shape)
        for pol in range(dirty["pixels"].data.shape[1]):
            if pol in (0, 3):
                if psf["pixels"].data[0, pol, :, :].max():
                    log.info(
                        "complex_hogbom_kernel_list: "
                        "Processing pol %d, channel %d",
                        pol,
                        channel,
                    )
                    window_data = (
                        window["pixels"].data[0, pol, :, :]
                        if window is not None
                        else None
                    )
                    (
                        comp_array[channel, pol, :, :],
                        residual_array[channel, pol, :, :],
                    ) = hogbom(
                        dirty["pixels"].data[0, pol, :, :],
                        psf["pixels"].data[0, pol, :, :],
                        window_data,
                        gain,
                        thresh,
                        niter,
                        fracthresh,
                    )
                else:
                    log.info(
                        "complex_hogbom_kernel_list: "
                        "Skipping pol %d, channel %d",
                        pol,
                        channel,
                    )
            if pol == 1:
                if psf["pixels"].data[0, 1:2, :, :].max():
                    log.info(
                        "complex_hogbom_kernel_list: "
                        "Processing pol 1 and 2, channel %d",
                        channel,
                    )
                    window_data = (
                        window["pixels"].data[0, pol, :, :]
                        if window is not None
                        else None
                    )
                    (
                        comp_array[channel, 1, :, :],
                        comp_array[channel, 2, :, :],
                        residual_array[channel, 1, :, :],
                        residual_array[channel, 2, :, :],
                    ) = hogbom_complex(
                        dirty["pixels"].data[0, 1, :, :],
                        dirty["pixels"].data[0, 2, :, :],
                        psf["pixels"].data[0, 1, :, :],
                        psf["pixels"].data[0, 2, :, :],
                        window_data,
                        gain,
                        thresh,
                        niter,
                        fracthresh,
                    )
                else:
                    log.info(
                        "complex_hogbom_kernel_list: "
                        "Skipping pol 1 and 2, channel %d",
                        channel,
                    )
                if pol == 2:
                    continue
        comp_image = Image.constructor(
            data=comp_array,
            polarisation_frame=PolarisationFrame("stokesIQUV"),
            wcs=dirty.image_acc.wcs,
        )
        residual_image = Image.constructor(
            data=residual_array,
            polarisation_frame=PolarisationFrame("stokesIQUV"),
            wcs=dirty.image_acc.wcs,
        )
        comp_images.append(comp_image)
        residual_images.append(residual_image)

    return comp_images, residual_images


def common_arguments(**kwargs):
    """Extract the common arguments from kwargs.

    :param gain: loop gain (float) default: 0.7
    :param niter: Number of minor cycle iterations: 100
    :param threshold: Clean threshold default 0.0
    :param fractional_threshold: Fractional threshold default 0.1
    :param scales: Scales (in pixels) for multiscale ([0, 3, 10, 30])

    :param kwargs:
    :return: fracthresh, gain, niter, thresh, scales
    """
    gain = kwargs.get("gain", 0.1)
    if gain <= 0.0 or gain >= 2.0:
        raise ValueError("Loop gain must be between 0 and 2")
    thresh = kwargs.get("threshold", 0.0)
    if thresh < 0.0:
        raise ValueError("Threshold must be positive or zero")
    niter = kwargs.get("niter", 100)
    if niter < 0:
        raise ValueError("niter must be greater than zero")
    fracthresh = kwargs.get("fractional_threshold", 0.01)
    if fracthresh < 0.0 or fracthresh > 1.0:
        raise ValueError("Fractional threshold should be in range 0.0, 1.0")
    scales = kwargs.get("scales", [0, 3, 10, 30])

    return fracthresh, gain, niter, thresh, scales


def hogbom_kernel_list(
    dirty_list: List[Image],
    prefix,
    psf_list: List[Image],
    window_list: List[Image],
    **kwargs,
):
    """
    Hogbom Clean, operating of lists of single frequency images.

    See: Hogbom CLEAN A&A Suppl, 15, 417, (1974)

    :param dirty_list: List of dirty images
    :param prefix: Informational string to be used in log messages
                e.g. "cycle 1, subimage 42"
    :param psf_list: List of Point Spread Function
    :param window_list: List of window images
    :param gain: loop gain (float) 0.1
    :param threshold: Clean threshold (0.0)
    :param fractional_threshold: Fractional threshold (0.01)
    :param scales: Scales (in pixels) for multiscale ([0, 3, 10, 30])
    :param nmoment: Number of frequency moments (default 3)
    :param findpeak: Method of finding peak in mfsclean:
                'Algorithm1'|'ASKAPSoft'|'CASA'|'RASCIL', Default is RASCIL.

    :return: component image_list, residual image_list
    """

    log.info(
        "hogbom_kernel_list %s: Starting Hogbom clean of "
        "each polarisation and channel separately",
        prefix,
    )

    fracthresh, gain, niter, thresh, _ = common_arguments(**kwargs)

    comp_images = []
    residual_images = []

    for channel, dirty in enumerate(dirty_list):
        psf = psf_list[channel]
        comp_array = numpy.zeros(dirty["pixels"].data.shape)
        residual_array = numpy.zeros(dirty["pixels"].data.shape)
        for pol in range(dirty["pixels"].data.shape[1]):
            if psf["pixels"].data[0, pol, :, :].max():
                log.info(
                    "hogbom_kernel_list %s: Processing pol %d, channel %d",
                    prefix,
                    pol,
                    channel,
                )
                if window_list is None or window_list[channel] is None:
                    (
                        comp_array[0, pol, :, :],
                        residual_array[0, pol, :, :],
                    ) = hogbom(
                        dirty["pixels"].data[0, pol, :, :],
                        psf["pixels"].data[0, pol, :, :],
                        None,
                        gain,
                        thresh,
                        niter,
                        fracthresh,
                        prefix,
                    )
                else:
                    (
                        comp_array[0, pol, :, :],
                        residual_array[0, pol, :, :],
                    ) = hogbom(
                        dirty["pixels"].data[0, pol, :, :],
                        psf["pixels"].data[0, pol, :, :],
                        window_list[channel]["pixels"].data[0, pol, :, :],
                        gain,
                        thresh,
                        niter,
                        fracthresh,
                        prefix,
                    )
            else:
                log.info(
                    "hogbom_kernel_list %s: Skipping pol %d, channel %d",
                    prefix,
                    pol,
                    channel,
                )
        comp_image = Image.constructor(
            data=comp_array,
            polarisation_frame=dirty.image_acc.polarisation_frame,
            wcs=dirty.image_acc.wcs,
        )
        residual_image = Image.constructor(
            data=residual_array,
            polarisation_frame=dirty.image_acc.polarisation_frame,
            wcs=dirty.image_acc.wcs,
        )
        comp_images.append(comp_image)
        residual_images.append(residual_image)

    return comp_images, residual_images


def mmclean_kernel_list(
    dirty_list: List[Image],
    prefix,
    psf_list: List[Image],
    window_list: List[Image] = None,
    sensitivity_list: List[Image] = None,
    **kwargs,
):
    """
    mfsmsclean, msmfsclean, mmclean: MultiScale Multi-Frequency CLEAN.

    See: U. Rau and T. J. Cornwell,
    “A multi-scale multi-frequency deconvolution algorithm
    for synthesis imaging in radio interferometry,” A&A 532, A71 (2011).

    For the MFS clean, the psf must have number of channels >= 2 * nmoment

    :param dirty_list: List of dirty images
    :param prefix: Informational string to be used in
            log messages e.g. "cycle 1, subimage 42"
    :param psf_list: List of Point Spread Function
    :param window_list: List of window images
    :param sensitivity_list: List of sensitivity images
    :return: component image_list, residual image_list

     The following optional arguments can be passed via kwargs:

    :param fractional_threshold: Fractional threshold (0.01)
    :param gain: loop gain (float) 0.7
    :param niter: Number of clean iterations (int) 100
    :param threshold: Clean threshold (0.0)
    :param scales: Scales (in pixels) for multiscale ([0, 3, 10, 30])
    :param nmoment: Number of frequency moments (default 3)
    :param findpeak: Method of finding peak in mfsclean:
                    'Algorithm1'|'CASA'|'RASCIL', Default is RASCIL.

    """

    findpeak = kwargs.get("findpeak", "RASCIL")

    log.info(
        "mmclean_kernel_list %s: "
        "Starting Multi-scale multi-frequency clean "
        "of each polarisation separately",
        prefix,
    )
    nmoment = kwargs.get("nmoment", 3)

    if not nmoment >= 1:
        raise ValueError(
            "Number of frequency moments must be greater than or equal to one"
        )

    fracthresh, gain, niter, thresh, scales = common_arguments(**kwargs)
    gain = kwargs.get("gain", 0.7)
    if not 0.0 < gain < 2.0:
        raise ValueError("Loop gain must be between 0 and 2")

    nchan = len(dirty_list)
    if not nchan > 2 * (nmoment - 1):
        raise ValueError(
            f"Requires `nchan > 2 * (nmoment - 1)` "
            f"({nchan} > {2 * (nmoment - 1)})"
        )
    dirty_taylor = calculate_image_list_frequency_moments(
        dirty_list, nmoment=nmoment
    )

    if sensitivity_list is not None:
        sensitivity_taylor = calculate_image_list_frequency_moments(
            sensitivity_list, nmoment=nmoment
        )
        sensitivity_taylor["pixels"].data /= nchan

    else:
        sensitivity_taylor = None

    if window_list is not None:
        window_taylor = calculate_image_list_frequency_moments(
            window_list, nmoment=nmoment
        )
        window_taylor["pixels"].data /= nchan
    else:
        window_taylor = None

    nmoment_for_psf = 2 * nmoment if nmoment > 1 else 1
    psf_taylor = calculate_image_list_frequency_moments(
        psf_list, nmoment=nmoment_for_psf
    )

    psf_peak = numpy.max(psf_taylor["pixels"].data)
    dirty_taylor["pixels"].data /= psf_peak
    psf_taylor["pixels"].data /= psf_peak
    log.info(
        "mmclean_kernel_list %s: Shape of Dirty moments image %s",
        prefix,
        str(dirty_taylor["pixels"].shape),
    )
    log.info(
        "mmclean_kernel_list %s: Shape of PSF moments image %s",
        prefix,
        str(psf_taylor["pixels"].shape),
    )

    comp_array = numpy.zeros(dirty_taylor["pixels"].data.shape)
    residual_array = numpy.zeros(dirty_taylor["pixels"].data.shape)
    for pol in range(dirty_taylor["pixels"].data.shape[1]):
        sens = (
            sensitivity_taylor["pixels"].data[:, pol, :, :]
            if sensitivity_taylor is not None
            else None
        )

        # Always use the moment 0, Stokes I PSF
        if psf_taylor["pixels"].data[0, 0, :, :].max():
            log.info("mmclean_kernel_list %s: Processing pol %d", prefix, pol)
            if window_taylor is None:
                (
                    comp_array[:, pol, :, :],
                    residual_array[:, pol, :, :],
                ) = msmfsclean(
                    dirty_taylor["pixels"].data[:, pol, :, :],
                    psf_taylor["pixels"].data[:, 0, :, :],
                    None,
                    sens,
                    gain,
                    thresh,
                    niter,
                    scales,
                    fracthresh,
                    findpeak,
                    prefix,
                )
            else:
                log.info(
                    "deconvolve_cube %s: Clean window has %d valid pixels",
                    prefix,
                    int(numpy.sum(window_taylor["pixels"].data[0, pol])),
                )
                (
                    comp_array[:, pol, :, :],
                    residual_array[:, pol, :, :],
                ) = msmfsclean(
                    dirty_taylor["pixels"].data[:, pol, :, :],
                    psf_taylor["pixels"].data[:, 0, :, :],
                    window_taylor["pixels"].data[0, pol, :, :],
                    sens,
                    gain,
                    thresh,
                    niter,
                    scales,
                    fracthresh,
                    findpeak,
                    prefix,
                )
        else:
            log.info("deconvolve_cube %s: Skipping pol %d", prefix, pol)
    comp_taylor = Image.constructor(
        data=comp_array,
        polarisation_frame=dirty_taylor.image_acc.polarisation_frame,
        wcs=dirty_taylor.image_acc.wcs,
    )
    residual_taylor = Image.constructor(
        data=residual_array,
        polarisation_frame=dirty_taylor.image_acc.polarisation_frame,
        wcs=dirty_taylor.image_acc.wcs,
    )
    log.info(
        "mmclean_kernel_list %s: calculating spectral "
        "image lists from frequency moment images",
        prefix,
    )
    comp_list = calculate_image_list_from_frequency_taylor_terms(
        dirty_list, comp_taylor
    )
    residual_list = calculate_image_list_from_frequency_taylor_terms(
        dirty_list, residual_taylor
    )
    return comp_list, residual_list


def msclean_kernel_list(
    dirty_list: List[Image],
    prefix,
    psf_list: List[Image],
    window_list: List[Image],
    sensitivity_list=None,
    **kwargs,
):
    """
    MultiScale CLEAN, operating of lists of single frequency images.

    See: Cornwell, T.J., Multiscale CLEAN (IEEE Journal of
    Selected Topics in Sig Proc, 2008 vol. 2 pp. 793-801)

    The clean search is performed on the product of the
    sensitivity image (if supplied) and the residual image.
    This gives a way to bias against high noise.

    :param dirty_list: List of dirty images
    :param prefix: Informational string to be used in
                log messages e.g. "cycle 1, subimage 42"
    :param psf_list: List of Point Spread Function
    :param window_list: List of window images
    :param sensitivity_list: List of sensitivity images
    :return: component image_list, residual image_list

     The following optional arguments can be passed via kwargs:

    :param fractional_threshold: Fractional threshold (0.01)
    :param gain: loop gain (float) 0.7
    :param niter: Number of clean iterations (int) 100
    :param threshold: Clean threshold (0.0)
    :param scales: Scales (in pixels) for multiscale ([0, 3, 10, 30])
    """
    log.info(
        "msclean_kernel_list %s: "
        "Starting Multi-scale clean of each "
        "polarisation and channel separately",
        prefix,
    )

    fracthresh, gain, niter, thresh, scales = common_arguments(**kwargs)

    comp_images = []
    residual_images = []

    for channel, dirty in enumerate(dirty_list):
        psf = psf_list[channel]

        comp_array = numpy.zeros_like(dirty["pixels"].data)
        residual_array = numpy.zeros_like(dirty["pixels"].data)

        for pol in range(dirty["pixels"].data.shape[1]):
            if (
                sensitivity_list is not None
                and sensitivity_list[channel] is not None
            ):
                sens = sensitivity_list[channel]["pixels"].data[0, pol, :, :]
            else:
                sens = None
            if psf["pixels"].data[0, pol, :, :].max():
                log.info(
                    "msclean_kernel_list %s: Processing pol %d, channel %d",
                    prefix,
                    pol,
                    channel,
                )
                if window_list is None or window_list[channel] is None:
                    (
                        comp_array[0, pol, :, :],
                        residual_array[0, pol, :, :],
                    ) = msclean(
                        dirty["pixels"].data[0, pol, :, :],
                        psf["pixels"].data[0, pol, :, :],
                        None,
                        sens,
                        gain,
                        thresh,
                        niter,
                        scales,
                        fracthresh,
                        prefix,
                    )
                else:
                    (
                        comp_array[0, pol, :, :],
                        residual_array[0, pol, :, :],
                    ) = msclean(
                        dirty["pixels"].data[0, pol, :, :],
                        psf["pixels"].data[0, pol, :, :],
                        window_list[channel]["pixels"].data[0, pol, :, :],
                        sens,
                        gain,
                        thresh,
                        niter,
                        scales,
                        fracthresh,
                        prefix,
                    )
            else:
                log.info(
                    "msclean_kernel_list %s: Skipping pol %d, channel %d",
                    prefix,
                    pol,
                    channel,
                )
        comp_image = Image.constructor(
            data=comp_array,
            polarisation_frame=dirty.image_acc.polarisation_frame,
            wcs=dirty.image_acc.wcs,
        )
        residual_image = Image.constructor(
            data=residual_array,
            polarisation_frame=dirty.image_acc.polarisation_frame,
            wcs=dirty.image_acc.wcs,
        )
        comp_images.append(comp_image)
        residual_images.append(residual_image)

    return comp_images, residual_images


def restore_list(
    model_list: List[Image],
    psf_list: List[Image] = None,
    residual_list: List[Image] = None,
    clean_beam=None,
):
    """Restore the model image to the residuals.

    The clean beam can be specified as a dictionary with
    fields "bmaj", "bmin" (both in arcsec) and "bpa" in degrees.

    :param model_list: Model image list (i.e. deconvolved)
    :param psf_list: Input PSF list (nchan)
    :param residual_list: Residual image
    :param clean_beam: Clean beam
                        e.g. {"bmaj":0.1, "bmin":0.05, "bpa":-60.0}.
                        Units are deg, deg, deg
    :return: restored_list

    """

    restored_list = []

    for channel, model in enumerate(model_list):
        if psf_list is not None:
            psf = psf_list[channel]
        else:
            psf = None

        restored = model.copy(deep=True)
        if residual_list is not None:
            residual = residual_list[channel]
        else:
            residual = None

        if clean_beam is None:
            if psf is not None:
                clean_beam = fit_psf(psf)
                log.info(
                    "restore_list: Using fitted clean beam "
                    "(deg, deg, deg) = %s",
                    clean_beam,
                )
            else:
                raise ValueError(
                    "restore_list: Either the psf or the "
                    "clean_beam must be specified"
                )
        else:
            log.info(
                "restore_list: Using clean beam (deg, deg, deg) = %s",
                (
                    clean_beam["bmaj"],
                    clean_beam["bmin"],
                    clean_beam["bpa"],
                ),
            )
            log.info(
                "restore_list: Using clean beam (arsec, arcsec, deg) = %s",
                (
                    3600.0 * clean_beam["bmaj"],
                    3600.0 * clean_beam["bmin"],
                    clean_beam["bpa"],
                ),
            )

        beam_pixels = convert_clean_beam_to_pixels(model, clean_beam)

        gk = Gaussian2DKernel(
            x_stddev=beam_pixels[0],
            y_stddev=beam_pixels[1],
            theta=beam_pixels[2],
        )
        # After the bug fix of astropy>=5.22, needs to normalize 'peak'
        gk.normalize(mode="peak")

        for chan in range(model["pixels"].shape[0]):
            for pol in range(model["pixels"].shape[1]):
                restored["pixels"].data[chan, pol, :, :] = convolve_fft(
                    model["pixels"].data[chan, pol, :, :],
                    gk,
                    normalize_kernel=False,
                    allow_huge=True,
                    boundary="wrap",
                )
        if residual is not None:
            restored["pixels"].data += residual["pixels"].data

        restored["pixels"].data = restored["pixels"].data.astype("float")

        restored.attrs["clean_beam"] = clean_beam

        restored_list.append(restored)

    return restored_list


def deconvolve_cube(
    dirty: Image, psf: Image, sensitivity: Image = None, prefix="", **kwargs
) -> (Image, Image):
    """Clean using a variety of algorithms.

    The algorithms available are:

     hogbom: Hogbom CLEAN See: Hogbom CLEAN A&A Suppl, 15, 417, (1974)

     hogbom-complex: Complex Hogbom CLEAN of stokesIQUV image

     msclean: MultiScale CLEAN See: Cornwell, T.J.,
     Multiscale CLEAN (IEEE Journal of Selected Topics in Sig Proc,
     2008 vol. 2 pp. 793-801)

     mfsmsclean, msmfsclean, mmclean: MultiScale Multi-Frequency
     See: U. Rau and T. J. Cornwell,
     “A multi-scale multi-frequency deconvolution algorithm
     for synthesis imaging in radio interferometry,” A&A 532, A71 (2011).

    For example::

        comp, residual = deconvolve_cube(dirty, psf, niter=1000,
                           gain=0.7, algorithm='msclean',
                           scales=[0, 3, 10, 30], threshold=0.01)

    For the MFS clean, the psf must have number of channels >= 2 * nmoment.

    :param dirty: Image dirty image
    :param psf: Image Point Spread Function
    :param sensitivity: Sensitivity image (i.e. inverse noise level)
    :param prefix: Informational message for logging
    :param window_shape: Window image (Bool) - clean where True
    :param mask: Window in the form of an image, overrides window_shape
    :param algorithm: Cleaning algorithm:
                'msclean'|'hogbom'|'hogbom-complex'|'mfsmsclean'
    :param gain: loop gain (float) 0.7
    :param threshold: Clean threshold (0.0)
    :param fractional_threshold: Fractional threshold (0.01)
    :param scales: Scales (in pixels) for multiscale ([0, 3, 10, 30])
    :param nmoment: Number of frequency moments (default 3)
    :param findpeak: Method of finding peak in mfsclean:
                    'Algorithm1'|'ASKAPSoft'|'CASA'|'RASCIL',
                    Default is RASCIL.
    :return: component image, residual image

     See also
        :py:func:`ska_sdp_func_python.image.cleaners.hogbom`
        :py:func:`ska_sdp_func_python.image.cleaners.hogbom_complex`
        :py:func:`ska_sdp_func_python.image.cleaners.msclean`
        :py:func:`ska_sdp_func_python.image.cleaners.msmfsclean`

    """
    dirty_list = image_scatter_channels(dirty)
    psf_list = image_scatter_channels(psf)
    if sensitivity is not None:
        sensitivity_list = image_scatter_channels(sensitivity)
    else:
        sensitivity_list = None

    comp_list, residual_list = deconvolve_list(
        dirty_list,
        psf_list,
        sensitivity=sensitivity_list,
        prefix=prefix,
        **kwargs,
    )
    comp = image_gather_channels(comp_list)
    residual = image_gather_channels(residual_list)
    return comp, residual


def fit_psf(psf: Image):
    """Fit a two-dimensional Gaussian to a PSF using astropy.modeling.

    :params psf: Input PSF
    :return: bmaj (arcsec), bmin (arcsec), bpa (deg)
    """
    npixel = psf["pixels"].data.shape[3]
    sl = slice(npixel // 2 - 7, npixel // 2 + 8)
    y, x = numpy.mgrid[sl, sl]
    z = psf["pixels"].data[0, 0, sl, sl]

    # isotropic at the moment!
    try:
        p_init = models.Gaussian2D(
            amplitude=numpy.max(z), x_mean=numpy.mean(x), y_mean=numpy.mean(y)
        )
        fit_p = fitting.LevMarLSQFitter()
        with warnings.catch_warnings():
            # Ignore model linearity warning from the fitter
            warnings.simplefilter("ignore")
            fit = fit_p(p_init, x, y, z)
        if fit.x_stddev <= 0.0 or fit.y_stddev <= 0.0:
            log.warning(
                "fit_psf: error in fitting to psf, using 1 pixel stddev"
            )
            beam_pixels = (1.0, 1.0, 0.0)
        else:
            # Note that the order here is minor, major, pa
            beam_pixels = (
                fit.x_stddev.value,
                fit.y_stddev.value,
                fit.theta.value,
            )
    except minpack.error:
        log.warning("fit_psf: minpack error, using 1 pixel stddev")
        beam_pixels = (1.0, 1.0, 0.0)
    except ValueError:
        log.warning("fit_psf: warning in fit to psf, using 1 pixel stddev")
        beam_pixels = (1.0, 1.0, 0.0)

    return convert_clean_beam_to_degrees(psf, beam_pixels)


def restore_cube(
    model: Image, psf=None, residual=None, clean_beam=None
) -> Image:
    """Restore the model image to the residuals.

    The clean beam can be specified as a dictionary with
    fields "bmaj", "bmin" (both in arcsec) and "bpa" in degrees.

    :param model: Model image (i.e. deconvolved)
    :param psf: Input PSF
    :param residual: Residual Image
    :param clean_beam: Clean beam e.g. {"bmaj":0.1, "bmin":0.05, "bpa":-60.0}.
                        Units are deg, deg, deg
    :return: restored image

    """
    model_list = image_scatter_channels(model)
    residual_list = image_scatter_channels(residual)
    psf_list = image_scatter_channels(psf)

    restored_list = restore_list(
        model_list=model_list,
        psf_list=psf_list,
        residual_list=residual_list,
        clean_beam=clean_beam,
    )

    return image_gather_channels(restored_list)
