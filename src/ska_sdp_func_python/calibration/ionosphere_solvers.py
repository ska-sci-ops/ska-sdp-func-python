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
    [param, coeff] = set_coeffs_and_params(xyz, cluster_id)

    n_param = get_param_count(param)[0]

    log.info(
        "Setting up iono solver for %d stations in %d cluster",
        len(gain_table.antenna),
        n_cluster,
    )
    if n_cluster == 1:
        log.info("There are %d total parameters in a single cluster", n_param)
    else:
        log.info(
            "There are %d total parameters: %d in c[0] + %d x c[1:%d]",
            n_param,
            len(param[0]),
            len(param[1]),
            len(param) - 1,
        )

    for it in range(niter):
        [AA, Ab] = build_normal_equation(
            vis, modelvis, param, coeff, cluster_id
        )

        # Solve the normal equations and update parameters
        param_update = solve_normal_equation(AA, Ab, param, it)

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

    :param: cluster_id
    :return n_cluster: total number of clusters
    :return cid2stn: mapping from station index to cluster index
    :return stn2cid: mapping from cluster index to a list of station indices

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
):
    """
    Calculate coefficients (a basis function value vector for each cluster) and
    initialise parameter values (a solution vector for each station)

    :param vis: Visibility containing the observed data_model
    :param xyz: [n_antenna,3] array containing the antenna locations in the
        local horizontal frame
    :param cluster_id:
    :return param: [n_cluster] list of solution vectors
    :return coeff: [n_station] list of basis-func value vectors
        Stored as a numpy dtype=object array of variable-length coeff vectors

    """
    # Get common mapping vectors between stations and clusters
    [n_cluster, cid2stn, _] = set_cluster_maps(cluster_id)

    n_station = len(cluster_id)
    coeff = [None] * n_station
    param = [None] * n_cluster

    # Coefficients and parameter values
    # treat cluster zero differently; it is assumed to be a larger central core
    cid = 0

    # Get Zernike parameters for the stations in this cluster
    zern_params = zern_array(
        6, xyz[cid2stn[cid], 0], xyz[cid2stn[cid], 1], noll_order=False
    )

    for idx, stn in enumerate(cid2stn[cid]):
        coeff[stn] = zern_params[idx]

    if len(cid2stn[cid]) > 0:
        param[cid] = numpy.zeros(len(coeff[cid2stn[cid][0]]))

    # now do the rest of the clusters
    for cid in range(1, n_cluster):
        # Remove the average position of the cluster
        xave = numpy.mean(xyz[cid2stn[cid], 0])
        yave = numpy.mean(xyz[cid2stn[cid], 1])
        for stn in cid2stn[cid]:
            # coeff[stn] = numpy.array([1, x[stn], y[stn]])
            coeff[stn] = numpy.array(
                [
                    1,
                    xyz[stn, 0] - xave,
                    xyz[stn, 1] - yave,
                ]
            )
        if len(cid2stn[cid]) > 0:
            param[cid] = numpy.zeros(len(coeff[cid2stn[cid][0]]))

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
    :param cluster_id:

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
    :param cluster_id:

    """
    # Get common mapping vectors between stations and clusters
    [n_cluster, _, stn2cid] = set_cluster_maps(cluster_id)
    [n_param, pidx0] = get_param_count(param)

    # set up a few references and constants
    frequency = vis.frequency.data
    wl_const = 2.0 * numpy.pi * const.c.value / frequency
    ant1 = vis.antenna1.data
    ant2 = vis.antenna2.data
    vis_data = vis.vis.data
    mdl_data = modelvis.vis.data

    n_baselines = len(vis.baselines)

    # exclude auto-correlations from the mask
    mask0 = ant1 != ant2

    # Loop over frequency and accumulate normal equations
    # Could probably handly frequency within an einsum as well.
    # It is also a natural axis for parallel calculation of AA and Ab.

    AA = numpy.zeros((n_param, n_param))
    Ab = numpy.zeros(n_param)

    blidx = numpy.arange(n_baselines)

    for chan in range(len(frequency)):
        # Could accumulate AA and Ab directly, but go via a
        # design matrix for clarity. Update later if need be.

        A = numpy.zeros((n_param, n_baselines), "complex_")

        # V = M * exp(i * 2*pi * wl * fit)
        # imag(V*conj(M)) = imag(|M|^2 * exp(i * 2*pi * wl * fit))
        #                 ~ |M|^2 * 2*pi * wl * fit
        # real(M*conj(M)) = |M|^2

        # Loop over pairs of clusters and update the design matrix for the
        # associated baselines
        for cid in range(0, n_cluster):
            pidx = numpy.arange(pidx0[cid], pidx0[cid] + len(param[cid]))

            # A mask for all baselines with ant1 in this cluster
            mask = mask0 * (stn2cid[ant1] == cid)
            if numpy.sum(mask) > 0:
                # [nvis] A0 terms x [nvis,nparam] coeffs (1st antenna)
                # all masked antennas have the same number of coeffs so can
                # form a coeff matrix and multiply
                ii, jj = numpy.meshgrid(pidx, blidx[mask], indexing="ij")
                A[ii, jj] += numpy.einsum(
                    "b,bp->pb",
                    wl_const[chan] * mdl_data[0, mask, chan, 0],
                    numpy.vstack(coeff[ant1[mask]]).astype("float_"),
                )

            # A mask for all baselines with ant2 in this cluster
            mask = mask0 * (stn2cid[ant2] == cid)
            if numpy.sum(mask) > 0:
                # [nvis] A0 terms x [nvis,nparam] coeffs (2nd antenna)
                ii, jj = numpy.meshgrid(pidx, blidx[mask], indexing="ij")
                A[ii, jj] -= numpy.einsum(
                    "b,bp->pb",
                    wl_const[chan] * mdl_data[0, mask, chan, 0],
                    numpy.vstack(coeff[ant2[mask]]).astype("float_"),
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
    :param cluster_id:

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
