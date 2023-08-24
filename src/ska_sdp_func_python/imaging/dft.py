"""
Functions that aid Fourier transform processing.
"""

__all__ = [
    "dft_cpu_looped",
    "dft_gpu_raw_kernel",
    "dft_kernel",
    "dft_skycomponent_visibility",
    "extract_direction_and_flux",
    "idft_visibility_skycomponent",
]

import collections
import logging
from typing import List, Union

import numpy
from scipy import interpolate
from ska_sdp_datamodels.science_data_model.polarisation_functions import (
    convert_pol_frame,
)
from ska_sdp_datamodels.sky_model.sky_model import SkyComponent
from ska_sdp_datamodels.visibility.vis_model import Visibility

from ska_sdp_func_python.util.coordinate_support import skycoord_to_lmn
from ska_sdp_func_python.visibility.base import calculate_visibility_phasor

log = logging.getLogger("func-python-logger")


def dft_skycomponent_visibility(
    vis: Visibility,
    sc: Union[SkyComponent, List[SkyComponent]],
    dft_compute_kernel=None,
) -> Visibility:
    """DFT to get the visibility from a SkyComponent, for Visibility.

    :param vis: Visibility
    :param sc: SkyComponent or list of SkyComponents
    :param dft_compute_kernel: string: cpu_looped, gpu_cupy_raw or proc_func
    :return: Visibility
    """
    if sc is None or (isinstance(sc, list) and len(sc) == 0):
        return vis

    direction_cosines, vfluxes = extract_direction_and_flux(sc, vis)

    vis["vis"].data = dft_kernel(
        direction_cosines,
        vfluxes,
        vis.visibility_acc.uvw_lambda,
        dft_compute_kernel=dft_compute_kernel,
    )

    return vis


def extract_direction_and_flux(sc, vis):
    """
    Extract SkyComponent direction and flux to be consumed by DFT.
    Flux polarisation and frequency are scaled to that
    of input Visibility data.

    :param sc: SkyComponent or list of SkyComponents
    :param vis: Visibility
    :returns: Tuple of two numpy arrays: component
              direction cosines and component fluxes
    """
    if not isinstance(sc, collections.abc.Iterable):
        sc = [sc]

    vfluxes = []  # Flux for each component
    direction_cosines = []  # lmn vector for each component

    for comp in sc:
        flux = comp.flux
        if comp.polarisation_frame != vis.visibility_acc.polarisation_frame:
            flux = convert_pol_frame(
                flux,
                comp.polarisation_frame,
                vis.visibility_acc.polarisation_frame,
            )

        # Interpolate in frequency if necessary
        if len(comp.frequency) == len(vis.frequency) and numpy.allclose(
            comp.frequency, vis.frequency.data, rtol=1e-15
        ):
            vflux = flux
        else:
            nchan, npol = flux.shape
            nvchan = len(vis.frequency)
            vflux = numpy.zeros([nvchan, npol])
            if nchan > 1:
                # TODO: this path needs testing to verify that it works
                for pol in range(flux.shape[1]):
                    fint = interpolate.interp1d(
                        comp.frequency, comp.flux[:, pol], kind="cubic"
                    )
                    vflux[:, pol] = fint(vis.frequency.data)
            else:
                # Just take the value since we cannot interpolate.
                # Might want to put some test here
                vflux = flux

        vfluxes.append(vflux)

        l, m, _ = skycoord_to_lmn(comp.direction, vis.phasecentre)
        direction_cosine = numpy.array(
            [l, m, numpy.sqrt(1 - l**2 - m**2) - 1.0]
        )

        direction_cosines.append(direction_cosine)

    direction_cosines = numpy.array(direction_cosines)
    vfluxes = numpy.array(vfluxes).astype("complex")

    return direction_cosines, vfluxes


def dft_kernel(
    direction_cosines, vfluxes, uvw_lambda, dft_compute_kernel=None
):
    """
    CPU computational kernel for DFT,
    choice dependent on dft_compute_kernel.

    :param direction_cosines: Direction cosines [ncomp, 3]
    :param vfluxes: Fluxes [ncomp, nchan, npol]
    :param uvw_lambda: UVW in lambda [ntimes, nbaselines, nchan, 3]
    :param dft_compute_kernel: string: cpu_looped, gpu_cupy_raw or proc_func
    :return: Vis array, dims: [ntimes, nbaselines, nchan, npol]
    """
    try:
        from ska_sdp_func.visibility import (
            dft_point_v00,  # pylint: disable=import-outside-toplevel
        )
    except ModuleNotFoundError as err:
        raise ModuleNotFoundError("ska_sdp_func is not installed!") from err

    if dft_compute_kernel is None:
        dft_compute_kernel = "cpu_looped"

    if dft_compute_kernel == "gpu_cupy_raw":
        return dft_gpu_raw_kernel(direction_cosines, uvw_lambda, vfluxes)
    if dft_compute_kernel == "cpu_looped":
        return dft_cpu_looped(direction_cosines, uvw_lambda, vfluxes)
    if dft_compute_kernel == "proc_func":
        # The Processing Function Library DFT function can be found at :
        # https://gitlab.com/ska-telescope/sdp/ska-sdp-func/-/blob/main/src/ska_sdp_func/dft.py

        log.info("Running with Processing Function Library DFT")
        if vfluxes.shape[1] == 1:
            # if sc is for a single channel, extract_direction_and_flux
            # will return fluxes for a single channel too;
            # this will break DFT if bvis is for multiple channels;
            # here we broadcast vfluxes to have the correct shape that
            # matches with the bvis.
            # Note: this is not needed for the func_python DFT, because numpy
            # correctly broadcasts the shapes at the place where its needed.
            comp_flux = numpy.ones(
                (vfluxes.shape[0], len(vfluxes.frequency), vfluxes.shape[-1]),
                dtype=complex,
            )
            comp_flux[:, :, :] = vfluxes
        else:
            comp_flux = vfluxes

        ntimes, nbaselines, nchan, _ = uvw_lambda.shape
        npol = vfluxes.shape[-1]
        new_vis_data = numpy.zeros(
            [ntimes, nbaselines, nchan, npol], dtype="complex"
        )

        dft_point_v00(
            direction_cosines,
            comp_flux,
            uvw_lambda,
            new_vis_data,
        )

        return new_vis_data

    raise ValueError(f"dft_compute_kernel {dft_compute_kernel} not known")


cuda_kernel_source = r"""
#include <cupy/complex.cuh>

#ifndef M_PI
#define M_PI 3.14159265358979323846264338327950288
#endif

#define INDEX_3D(N3, N2, N1, I3, I2, I1)         (N1 * (N2 * I3 + I2) + I1)
#define INDEX_4D(N4, N3, N2, N1, I4, I3, I2, I1) (N1 * (N2 * (N3 * I4 + I3) + I2) + I1)

extern "C" {

__global__ void dft_kernel(
        const int num_components,
        const int num_pols,
        const int num_channels,
        const int num_baselines,
        const int num_times,
        const double3         *const __restrict__ direction_cosines,        // Source direction cosines [num_components]
        const complex<double> *const __restrict__ vfluxes,    // Source fluxes [num_components, num_channels, num_pols]
        const double3         *const __restrict__ uvw_lambda, // UVW in lambda [num_times, num_baselines, num_channels]
        complex<double>       *__restrict__ vis)              // Visibilities  [num_times, num_baselines, num_channels, num_pols]
{
    // Local (per-thread) visibility.
    complex<double> vis_local[4]; // Allow up to 4 polarisations.
    vis_local[0] = vis_local[1] = vis_local[2] = vis_local[3] = 0.0;

    // Get indices of the output array this thread is working on.
    const int i_baseline = blockDim.x * blockIdx.x + threadIdx.x;
    const int i_channel  = blockDim.y * blockIdx.y + threadIdx.y;
    const int i_time     = blockDim.z * blockIdx.z + threadIdx.z;

    // Bounds check.
    if (num_pols > 4 ||
            i_baseline >= num_baselines ||
            i_channel >= num_channels ||
            i_time >= num_times) {
        return;
    }

    // Load uvw-coordinates.
    const double3 uvw = uvw_lambda[INDEX_3D(
            num_times, num_baselines, num_channels,
            i_time, i_baseline, i_channel)];

    // Loop over components and calculate phase for each.
    for (int i_component = 0; i_component < num_components; ++i_component) {
        double sin_phase, cos_phase;
        const double3 dir = direction_cosines[i_component];
        const double phase = -2.0 * M_PI * (
                dir.x * uvw.x + dir.y * uvw.y + dir.z * uvw.z);
        sincos(phase, &sin_phase, &cos_phase);
        complex<double> phasor(cos_phase, sin_phase);

        // Multiply by flux in each polarisation and accumulate.
        const unsigned int i_pol_start = INDEX_3D(
                num_components, num_channels, num_pols,
                i_component, i_channel, 0);
        if (num_pols == 1) {
            vis_local[0] += (phasor * vfluxes[i_pol_start]);
        } else if (num_pols == 4) {
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                vis_local[i] += (phasor * vfluxes[i_pol_start + i]);
            }
        }
    }

    // Write out local visibility.
    for (int i = 0; i < num_pols; ++i) {
        const unsigned int i_out = INDEX_4D(num_times, num_baselines,
                num_channels, num_pols, i_time, i_baseline, i_channel, i);
        vis[i_out] = vis_local[i];
    }
}

}
"""  # noqa: E501


def dft_cpu_looped(direction_cosines, uvw_lambda, vfluxes):
    """CPU computational kernel for DFT, using explicit loop over components.

    :param direction_cosines: Direction cosines [ncomp, 3]
    :param vfluxes: Fluxes [ncomp, nchan, npol]
    :param uvw_lambda: UVW in lambda [ntimes, nbaselines, nchan, 3]
    :return: Vis array, dims: [ntimes, nbaselines, nchan, npol]
    """
    ncomp, _ = direction_cosines.shape
    ntimes, nbaselines, nchan, _ = uvw_lambda.shape
    npol = vfluxes.shape[-1]
    vis = numpy.zeros([ntimes, nbaselines, nchan, npol], dtype="complex")
    for icomp in range(ncomp):
        phasor = numpy.exp(
            -2j
            * numpy.pi
            * numpy.sum(uvw_lambda * direction_cosines[icomp, :], axis=-1)
        )
        for pol in range(npol):
            vis[..., pol] += vfluxes[icomp, :, pol] * phasor
    return vis


def dft_gpu_raw_kernel(direction_cosines, uvw_lambda, vfluxes):
    """CPU computational kernel for DFT, using CUDA raw code via cupy.

    :param direction_cosines: Direction cosines [ncomp, 3]
    :param vfluxes: Fluxes [ncomp, nchan, npol]
    :param uvw_lambda: UVW in lambda [ntimes, nbaselines, nchan, 3]
    :return: Vis array, dims: [ntimes, nbaselines, nchan, npol]
    """
    try:
        import cupy  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError as err:
        raise ModuleNotFoundError(
            "cupy is not installed - cannot run CUDA"
        ) from err

    # Get the dimension sizes.
    (num_times, num_baselines, num_channels, _) = uvw_lambda.shape
    (num_components, _, num_pols) = vfluxes.shape
    # Get a handle to the GPU kernel.
    module = cupy.RawModule(code=cuda_kernel_source)
    kernel_dft = module.get_function("dft_kernel")
    # Allocate GPU memory and copy input arrays.
    direction_cosines_gpu = cupy.asarray(direction_cosines)
    fluxes_gpu = cupy.asarray(vfluxes)
    uvw_gpu = cupy.asarray(uvw_lambda)
    vis_gpu = cupy.zeros(
        (num_times, num_baselines, num_channels, num_pols),
        dtype=cupy.complex128,
    )
    # Define GPU kernel parameters, thread block size and grid size.
    num_threads = (128, 2, 2)  # Product must not exceed 1024.
    num_blocks = (
        (num_baselines + num_threads[0] - 1) // num_threads[0],
        (num_channels + num_threads[1] - 1) // num_threads[1],
        (num_times + num_threads[2] - 1) // num_threads[2],
    )
    args = (
        num_components,
        num_pols,
        num_channels,
        num_baselines,
        num_times,
        direction_cosines_gpu,
        fluxes_gpu,
        uvw_gpu,
        vis_gpu,
    )
    # Call the GPU kernel and copy results to host.
    kernel_dft(num_blocks, num_threads, args)
    return cupy.asnumpy(vis_gpu)


def idft_visibility_skycomponent(
    vis: Visibility, sc: Union[SkyComponent, List[SkyComponent]]
) -> ([SkyComponent, List[SkyComponent]], List[numpy.ndarray]):
    """Inverse DFT a SkyComponent from Visibility.

    :param vis: Visibility
    :param sc: SkyComponent or list of SkyComponents
    :return: SkyComponent or list of SkyComponents, array of weights
    """
    if sc is None:
        return sc

    if not isinstance(sc, collections.abc.Iterable):
        sc = [sc]

    newsc = []
    weights_list = []

    for comp in sc:
        # assert isinstance(comp, SkyComponent), comp
        newcomp = comp.copy()

        phasor = numpy.conjugate(
            calculate_visibility_phasor(comp.direction, vis)
        )
        flux = numpy.sum(
            vis.visibility_acc.flagged_weight
            * vis.visibility_acc.flagged_vis
            * phasor,
            axis=(0, 1),
        )
        weight = numpy.sum(vis.visibility_acc.flagged_weight, axis=(0, 1))

        flux[weight > 0.0] = flux[weight > 0.0] / weight[weight > 0.0]
        flux[weight <= 0.0] = 0.0
        if comp.polarisation_frame != vis.visibility_acc.polarisation_frame:
            flux = convert_pol_frame(
                flux,
                vis.visibility_acc.polarisation_frame,
                comp.polarisation_frame,
            )

        newcomp.flux = flux

        newsc.append(newcomp)
        weights_list.append(weight)

    return newsc, weights_list
