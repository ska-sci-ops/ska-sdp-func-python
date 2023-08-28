# pylint: disable=invalid-name,too-many-arguments,no-member

"""
Functions to solve for delta-TEC variations across the array
"""

__all__ = ["solve_ionosphere"]

import logging

import numpy
from astropy import constants as const
from ska_sdp_datamodels.calibration.calibration_create import (
    create_gaintable_from_visibility,
)
from ska_sdp_datamodels.calibration.calibration_model import GainTable
from ska_sdp_datamodels.visibility.vis_model import Visibility

from ska_sdp_func_python.calibration.ionosphere_utils import zern_array

log = logging.getLogger("func-python-logger")


def solve_ionosphere(
    vis: Visibility,
    modelvis: Visibility,
    xyz,
    cluster_id=None,
    zernike_limit=None,
    block_diagonal=False,
    niter=15,
    tol=1e-6,
) -> GainTable:
    # pylint: disable=too-many-locals
    """
    Solve a gain table by fitting for delta-TEC variations across the array
    The resulting delta-TEC variations will be converted to antenna-dependent
    phase shifts and the gain_table updated.

    Fits are performed within user-defined station clusters

    TODO: phase referencing to reference antenna
    TODO: user (and/or auto) control of num param per cluster?
    TODO: check convergence WRT adaptation factor nu (~ 1-damping in solver.py)

    :param vis: Visibility containing the observed data_model
    :param modelvis: Visibility containing the predicted data_model
    :param xyz: [n_antenna,3] array containing the antenna locations in the
        local horizontal frame
    :param cluster_id: [n_antenna] array containing the cluster ID of each
        antenna. Defaults to a single cluster comprising all stations
    :param zernike_limit: [n_cluster] list of Zernike index limits.
        Default is to leave unset when calling set_coeffs_and_params().
    :param block_diagonal: If true, each cluster will be solver for separately
        during each iteration. This is equivalent to setting all elements of
        the normal matrix to zero except for the block diagonal elements for
        the cluster in question. (default False)
    :param niter: Number of iterations (default 15)
    :param tol: Iteration stops when the fractional change in the gain solution
        is below this tolerance.
    :return: GainTable containing solutions

    """
    if numpy.all(modelvis.vis == 0.0):
        raise ValueError("solve_ionosphere: Model visibilities are zero")

    # Create a new gaintable based on the visibilities
    # In general it will be filled with antenna-based phase shifts per channel
    gain_table = create_gaintable_from_visibility(vis, jones_type="B")

    # Ensure that the gain table and the input cluster indices are consistent
    if cluster_id is None:
        cluster_id = numpy.zeros(len(gain_table.antenna), "int")

    n_cluster = numpy.amax(cluster_id) + 1

    # Could be less strict & require max(gain_table.antenna) < len(cluster_id)
    if len(gain_table.antenna) != len(cluster_id):
        raise ValueError(f"cluster_id has wrong size {len(cluster_id)}")

    # Calculate coefficients for each cluster and initialise parameter values
    [param, coeff] = set_coeffs_and_params(xyz, cluster_id, zernike_limit)

    n_param = get_param_count(param)[0]

    if n_cluster == 1:
        log.info(
            "Setting up iono solver for %d stations in a single cluster",
            len(gain_table.antenna),
        )
        log.info("There are %d total parameters in the cluster", n_param)
    else:
        log.info(
            "Setting up iono solver for %d stations in %d clusters",
            len(gain_table.antenna),
            n_cluster,
        )
        log.info(
            "There are %d total parameters: %d in c[0] + %d x c[1:%d]",
            n_param,
            len(param[0]),
            len(param[1]),
            len(param) - 1,
        )

    for it in range(niter):
        if not block_diagonal:
            [AA, Ab] = build_normal_equation(
                vis, modelvis, param, coeff, cluster_id
            )

            # Solve the normal equations and update parameters
            param_update = solve_normal_equation(AA, Ab, param, it)

        else:
            i0 = 0
            param_update = []
            for cid in range(n_cluster):
                n_cparam = len(param[cid])

                [AA, Ab] = build_normal_equation(
                    vis, modelvis, param, coeff, cluster_id, cid
                )

                # Solve the current incremental normal equations
                soln_vec = numpy.linalg.lstsq(AA, Ab, rcond=None)[0]

                # Update factor
                nu = 0.5
                # nu = 1.0 - 0.5 * (it % 2)
                param_update.append(nu * soln_vec)
                param[cid] += param_update[cid]

                i0 += n_cparam

        # Update the model
        apply_phase_distortions(modelvis, param_update, coeff, cluster_id)

        # test absolute relative change against tol
        # flag for non-zero parameters to test relative change against
        mask = numpy.abs(numpy.hstack(param).astype("float_")) > 0.0
        change = numpy.max(
            numpy.abs(numpy.hstack(param_update)[mask].astype("float_"))
            / numpy.abs(numpy.hstack(param)[mask].astype("float_"))
        )
        if change < tol:
            break

    # Update and return the gain table
    update_gain_table(gain_table, param, coeff, cluster_id)

    return gain_table


def set_cluster_maps(cluster_id):
    """
    Return vectors to help convert between station and cluster indices

    :param: cluster_id: [n_antenna] array of antenna cluster indices
    :return n_cluster: total number of clusters
    :return cid2stn: mapping from cluster index to List of station indices
    :return stn2cid: mapping from station index to cluster index

    """
    n_station = len(cluster_id)
    n_cluster = numpy.amax(cluster_id) + 1

    # Mapping from station index to cluster index
    stn2cid = numpy.empty(n_station, "int")

    # Mapping from cluster index to a list of station indices
    cid2stn = []

    stations = numpy.arange(n_station).astype("int")
    for cid in range(n_cluster):
        mask = cluster_id == cid
        cid2stn.append(stations[mask])
        stn2cid[mask] = cid

    return n_cluster, cid2stn, stn2cid


def get_param_count(param):
    """
    Return the total number of parameters across all clusters and the starting
    index of each cluster in stacked parameter vectors

    :param param: [n_cluster] list of solution vectors, one for each cluster
    :return n_param: int, total number of parameters
    :return pidx0: [n_cluster], starting index of each cluster in param vectors

    """
    n_cluster = len(param)

    # Total number of parameters across all clusters
    n_param = 0

    # Starting parameter for each cluster
    pidx0 = numpy.zeros(n_cluster, "int")

    for cid in range(n_cluster):
        pidx0[cid] = n_param
        n_param += len(param[cid])

    return n_param, pidx0


def set_coeffs_and_params(
    xyz,
    cluster_id,
    zernike_limit=None,
):
    """
    Calculate coefficients (a basis function value vector for each cluster) and
    initialise parameter values (a solution vector for each station)

    :param xyz: [n_antenna,3] array containing the antenna locations in the
        local horizontal frame
    :param cluster_id: [n_antenna] array of antenna cluster indices
    :param zernike_limit: [n_cluster] list of Zernike index limits:
        n + |m| <= zernike_limit[cluster_id]. Default: [6,2,2,...,2]
    :return param: [n_cluster] list of solution vectors
    :return coeff: [n_station] list of basis-func value vectors
        Stored as a numpy dtype=object array of variable-length coeff vectors

    """
    # Get common mapping vectors between stations and clusters
    [n_cluster, cid2stn, _] = set_cluster_maps(cluster_id)

    n_station = len(cluster_id)
    coeff = [None] * n_station
    param = [None] * n_cluster

    # Check list of polynomial degree limits
    if zernike_limit is None:
        # set a TEC offset and ramp for most clusters
        zernike_limit = [2] * n_cluster
        # but assume cluster zero contains the large central core
        zernike_limit[0] = 6
    elif len(zernike_limit) != n_cluster:
        log.error("Incorrect length for zernike_limit parmater")
        return numpy.empty(0), numpy.empty(0)

    for cid in range(0, n_cluster):
        # Generate the required Zernike polynomials for each station
        zern_params = zern_array(
            zernike_limit[cid], xyz[cid2stn[cid], 0], xyz[cid2stn[cid], 1]
        )

        # Set coefficients
        for idx, stn in enumerate(cid2stn[cid]):
            coeff[stn] = zern_params[idx]

        # Initialise parameters
        if len(cid2stn[cid]) > 0:
            param[cid] = numpy.zeros(len(coeff[cid2stn[cid][0]]))

    # # Get Zernike parameters for stations in the larger central cluster
    # cid = 0
    # zern_params = zern_array(
    #     zernike_limit[cid], xyz[cid2stn[cid], 0], xyz[cid2stn[cid], 1]
    # )

    # for idx, stn in enumerate(cid2stn[cid]):
    #     coeff[stn] = zern_params[idx]

    # if len(cid2stn[cid]) > 0:
    #     param[cid] = numpy.zeros(len(coeff[cid2stn[cid][0]]))

    # # now do the rest of the clusters
    # for cid in range(1, n_cluster):
    #     # Remove the average position of the cluster
    #     xave = numpy.mean(xyz[cid2stn[cid], 0])
    #     yave = numpy.mean(xyz[cid2stn[cid], 1])
    #     for stn in cid2stn[cid]:
    #         # coeff[stn] = numpy.array([1, x[stn], y[stn]])
    #         coeff[stn] = numpy.array(
    #             [
    #                 1,
    #                 xyz[stn, 0] - xave,
    #                 xyz[stn, 1] - yave,
    #             ]
    #         )
    #     if len(cid2stn[cid]) > 0:
    #         param[cid] = numpy.zeros(len(coeff[cid2stn[cid][0]]))

    return param, numpy.array(coeff, dtype=object)


def apply_phase_distortions(
    vis: Visibility,
    param,
    coeff,
    cluster_id,
):
    """
    Update visibility model with new fit solutions

    :param vis: Visibility containing the data_models to be distorted
    :param param: [n_cluster] list of solution vectors, one for each cluster
    :param coeff: [n_station] list of basis-func value vectors, one per station
        Stored as a numpy dtype=object array of variable-length coeff vectors
    :param cluster_id: [n_antenna] array of antenna cluster indices

    """
    # Get common mapping vectors between stations and clusters
    [n_cluster, _, stn2cid] = set_cluster_maps(cluster_id)

    # set up a few references and constants
    ant1 = vis.antenna1.data
    ant2 = vis.antenna2.data
    vis_data = vis.vis.data

    # exclude auto-correlations from the mask
    mask0 = ant1 != ant2

    # Use einsum calls to average over parameters for all combinations of
    # baseline and frequency
    # [n_freq] scaling constants
    # Loop over pairs of clusters and update the associated baselines
    for cid1 in range(0, n_cluster):
        for cid2 in range(0, n_cluster):
            # A mask for all baselines in this cluster pair
            mask = mask0 * (stn2cid[ant1] == cid1) * (stn2cid[ant2] == cid2)
            if numpy.sum(mask) == 0:
                continue
            vis_data[0, mask, :, 0] *= numpy.exp(
                # combine parmas for [n_baseline] then scale for [n_freq]
                numpy.einsum(
                    "b,f->bf",
                    (
                        # combine parmas for ant i in baselines
                        numpy.einsum(
                            "bp,p->b",
                            numpy.vstack(coeff[ant1[mask]]).astype("float_"),
                            param[cid1],
                        )
                        # combine parmas for ant j in baselines
                        - numpy.einsum(
                            "bp,p->b",
                            numpy.vstack(coeff[ant2[mask]]).astype("float_"),
                            param[cid2],
                        )
                    ),
                    # phase scaling with frequency
                    1j * 2.0 * numpy.pi * const.c.value / vis.frequency.data,
                )
            )


def build_normal_equation(
    vis: Visibility,
    modelvis: Visibility,
    param,
    coeff,
    cluster_id,
    cid=None,
):
    # pylint: disable=too-many-locals
    """
    Build normal equations for the chosen parameters and the current model
    visibilties

    :param vis: Visibility containing the observed data_models
    :param modelvis: Visibility containing the predicted data_models
    :param param: [n_cluster] list of solution vectors, one for each cluster
    :param coeff: [n_station] list of basis-func value vectors, one per station
        Stored as a numpy dtype=object array of variable-length coeff vectors
    :param cluster_id: [n_antenna] array of antenna cluster indices
    :param cid: index of current cluster. Defaults to None, which will build
        a single large matrix for all clusters.

    """

    # If no cluster index is given, build matrix for all clusters
    generate_full_equation = cid is None

    # Get common mapping vectors between stations and clusters
    [n_cluster, _, stn2cid] = set_cluster_maps(cluster_id)

    # Set up a few refs/consts to use in loops and function calls
    wl_const = 2.0 * numpy.pi * const.c.value / vis.frequency.data
    ant1 = vis.antenna1.data
    ant2 = vis.antenna2.data
    vis_data = vis.vis.data
    mdl_data = modelvis.vis.data

    n_baselines = len(vis.baselines)

    # Exclude auto-correlations from the mask
    mask = ant1 != ant2

    # Loop over frequency and accumulate normal equations
    # Could probably handly frequency within an einsum as well.
    # It is also a natural axis for parallel calculation of AA and Ab.

    if generate_full_equation:
        [n_param, pidx0] = get_param_count(param)
    else:
        n_param = len(param[cid])

    AA = numpy.zeros((n_param, n_param))
    Ab = numpy.zeros(n_param)

    for chan in range(len(vis.frequency.data)):
        # Could accumulate AA and Ab directly, but go via a
        # design matrix for clarity. Update later if need be.

        # V = M * exp(i * 2*pi * wl * fit)
        # imag(V*conj(M)) = imag(|M|^2 * exp(i * 2*pi * wl * fit))
        #                 ~ |M|^2 * 2*pi * wl * fit
        # real(M*conj(M)) = |M|^2

        if generate_full_equation:
            A = numpy.zeros((n_param, n_baselines), "complex_")

            # Loop over clusters and update the design matrix for the
            # associated baselines
            for _cid in range(0, n_cluster):
                pid = numpy.arange(pidx0[_cid], pidx0[_cid] + len(param[_cid]))

                A[pid, :] += cluster_design_matrix(
                    mdl_data[0, :, chan, 0],
                    mask,
                    ant1,
                    ant2,
                    coeff,
                    stn2cid,
                    wl_const[chan],
                    len(param[_cid]),
                    _cid,
                )

        else:
            A = cluster_design_matrix(
                mdl_data[0, :, chan, 0],
                mask,
                ant1,
                ant2,
                coeff,
                stn2cid,
                wl_const[chan],
                n_param,
                cid,
            )

        # Average over all baselines for each param pair
        AA += numpy.real(numpy.einsum("pb,qb->pq", numpy.conj(A), A))
        Ab += numpy.imag(
            numpy.einsum(
                "pb,b->p",
                numpy.conj(A),
                vis_data[0, :, chan, 0] - mdl_data[0, :, chan, 0],
            )
        )

    return AA, Ab


def cluster_design_matrix(
    mdl_data,
    mask0,
    ant1,
    ant2,
    coeff,
    stn2cid,
    wl_const,
    n_param,
    cid,
) -> numpy.ndarray:
    """
    Generate elements of the design matrix for the current cluster

    Dereference outside of loops and the function call to avoid overheads

    :param mdl_data: [n_time,n_baseline,n_pol] predicted model vis for chan
    :param mask0: [n_baseline] mask of wanted data samples
    :param ant1: [n_baseline] station index of first antenna in each baseline
    :param ant2: [n_baseline] station index of second antenna in each baseline
    :param coeff: [n_station] list of basis-func value vectors, one per station
    :param stn2cid: [n_station] mapping from station index to cluster index
    :param wl_const: 2*pi*lambda for the current frequency channel
    :param cid: index of current cluster
    :param n_param: number of parameters in Normal equation

    """

    n_baselines = len(mask0)

    A = numpy.zeros((n_param, n_baselines), "complex_")

    blidx_all = numpy.arange(n_baselines)

    # Get all masked baselines with ant1 in this cluster
    blidx = blidx_all[mask0 * (stn2cid[ant1] == cid)]
    if len(blidx) > 0:
        # [nvis] A0 terms x [nvis,nparam] coeffs (1st antenna)
        # all masked antennas have the same number of coeffs so can vstack
        A[:, blidx] += numpy.einsum(
            "b,bp->pb",
            wl_const * mdl_data[blidx],
            numpy.vstack(coeff[ant1[blidx]]).astype("float_"),
        )

    # Get all masked baselines with ant2 in this cluster
    blidx = blidx_all[mask0 * (stn2cid[ant2] == cid)]
    if len(blidx) > 0:
        # [nvis] A0 terms x [nvis,nparam] coeffs (2nd antenna)
        # all masked antennas have the same number of coeffs so can vstack
        A[:, blidx] -= numpy.einsum(
            "b,bp->pb",
            wl_const * mdl_data[blidx],
            numpy.vstack(coeff[ant2[blidx]]).astype("float_"),
        )

    return A


def solve_normal_equation(
    AA,
    Ab,
    param,
    it=0,  # pylint: disable=unused-argument
):
    """
    Solve the normal equations and update parameters

    Using the SVD-based DGELSD solver via numpy.linalg.lstsq.
    Could use the LU-decomposition-based DGESV solver in numpy.linalg.solve,
    but the normal matrix may not be full rank.

    If n_param gets large (~ 100) it may be better to use a numerical solver
    like lsmr or lsqr.

    :param AA: [n_param, n_param] normal equation
    :param Ab: [n_param] data vector
    :param param: [n_cluster] list of solution vectors, one for each cluster
    :param it: int, current iteration
    :return param_update: the current incremental param update

    """
    n_cluster = len(param)
    [_, pidx0] = get_param_count(param)

    # Solve the current incremental normal equations
    soln_vec = numpy.linalg.lstsq(AA, Ab, rcond=None)[0]

    # Make a copy of coeff for just the current incremental update
    param_update = []
    for cid in range(n_cluster):
        param_update.append(numpy.zeros(len(param[cid])))

    # Update factor
    nu = 0.5
    # StefCal-like algorithms work well with an alternating factor like this
    # Some early tests of this algorithm did as well. Come back to this
    # nu = 1.0 - 0.5 * (it % 2)
    for cid in range(n_cluster):
        param_update[cid] = (
            nu
            * soln_vec[pidx0[cid] : pidx0[cid] + len(param[cid])]  # noqa: E203
        )
        param[cid] += param_update[cid]

    return param_update


def update_gain_table(
    gain_table: GainTable,
    param,
    coeff,
    cluster_id,
):
    """
    Expand solutions for all stations and frequency channels and insert in the
    gain table

    :param gain_table: GainTable to be updated
    :param param: [n_cluster] list of solution vectors, one for each cluster
    :param coeff: [n_station] list of basis-func value vectors, one per station
        Stored as a numpy dtype=object array of variable-length coeff vectors
    :param cluster_id: [n_antenna] array of antenna cluster indices

    """
    # Get common mapping vectors between stations and clusters
    [n_cluster, cid2stn, _] = set_cluster_maps(cluster_id)

    wl = const.c.value / gain_table.frequency.data

    table_data = gain_table.gain.data

    for cid in range(0, n_cluster):
        # combine parmas for [n_station] phase terms then scale for [n_freq]
        table_data[0, cid2stn[cid], :, 0, 0] = numpy.exp(
            numpy.einsum(
                "s,f->sf",
                numpy.einsum(
                    "sp,p->s",
                    numpy.vstack(coeff[cid2stn[cid]]).astype("float_"),
                    param[cid],
                ),
                1j * 2.0 * numpy.pi * wl,
            )
        )
