"""
Functions to solve for and apply chains of antenna/station gain tables.
See documentation for further information.
"""

__all__ = [
    "apply_calibration_chain",
    "calibrate_chain",
    "create_calibration_controls",
    "solve_calibrate_chain",
]

import logging

import numpy
from ska_sdp_datamodels.calibration.calibration_create import (
    create_gaintable_from_visibility,
)
from ska_sdp_datamodels.calibration.calibration_model import GainTable

from ska_sdp_func_python.calibration.operations import apply_gaintable
from ska_sdp_func_python.calibration.solvers import solve_gaintable

log = logging.getLogger("func-python-logger")


def create_calibration_controls():
    """
    Create a dictionary containing default chain calibration controls.

    The fields are

        T: Atmospheric phase
        G: Electronic gains
        B: Bandpass

    Not supported:
        P: Polarisation
        I: Ionosphere

    Therefore, first get this default dictionary and then
    adjust parameters as desired.
    The calibrate function takes a context string e.g. TGB.
    It then calibrates each of these Jones matrices in turn.

     Note that P and I calibration require off diagonal terms producing n
     on-commutation of the Jones matrices. This is not handled yet.

    :return: dictionary
    """
    controls = {
        "T": {
            "shape": "scalar",
            "timeslice": "auto",
            "phase_only": True,
            "first_selfcal": 0,
        },
        "G": {
            "shape": "vector",
            "timeslice": 60.0,
            "phase_only": False,
            "first_selfcal": 0,
        },
        "B": {
            "shape": "vector",
            "timeslice": 1e5,
            "phase_only": False,
            "first_selfcal": 0,
        },
    }

    return controls


def apply_calibration_chain(
    vis,
    gaintables,
    calibration_context="T",
    controls=None,
    iteration=0,
    inverse=True,
):
    """
    Update the Visibility using the calibrated solutions
    in the form of GainTables.

    The context string can denote a sequence of calibrations
    e.g. TGB.
    Currently, we do not support inputting different timescales.

    :param vis: Visibility
    :param gaintables: Calibrated gaintables
                       Can be a GainTable, a List or a Dict of GainTables
    :param calibration_context: calibration contexts in order
                    of correction e.g. 'TGB'
    :param controls: Controls dictionary, modified as necessary
    :param iteration: Iteration number to be compared
                    to the 'first_selfcal' field.
    :param inverse: The inverse operation of applying a gain table
                    to a visibility
    :return: Visibility after calibration solution applied
             Or return original visibility if the GainTables provided
             don't match the calibration context.
    """

    if controls is None:
        controls = create_calibration_controls()

    if isinstance(gaintables, GainTable):
        gaintables = [gaintables]

    # Check if the calibration contexts
    # match with the Jones types in the GainTables
    contexts = []
    gt = {}
    if isinstance(gaintables, list):
        for gaintable in gaintables:
            if gaintable.attrs["jones_type"] in list(calibration_context):
                contexts.append(gaintable.attrs["jones_type"])
                gt[gaintable.attrs["jones_type"]] = gaintable
    elif isinstance(gaintables, dict):
        gt = gaintables
        for context, _ in gaintables.items():
            contexts.append(context)

    else:
        log.warning(
            "Invalid GainTable format supplied. Visibility not updated."
        )
        return vis

    # Only apply if the context list is not empty
    # else return the original Visibility
    if contexts:
        for c in contexts:
            if iteration >= controls[c]["first_selfcal"]:
                vis = apply_gaintable(vis, gt[c], inverse=inverse)

    return vis


def calibrate_chain(
    vis,
    model_vis,
    gaintables=None,
    calibration_context="T",
    controls=None,
    iteration=0,
    tol=1e-6,
):
    """
    Calibrate using algorithm specified by calibration_context.

    The context string can denote a sequence of calibrations
    e.g. TGB.
    Currently, we do not support inputting different timescales.

    :param vis: Visibility containing the observed data_models
    :param model_vis: Visibility containing the visibility predicted by a model
    :param gaintables: Existing GainTables (GainTable, list or dict)
    :param calibration_context: Calibration contexts in order
                of correction e.g. 'TGB'
    :param controls: Controls dictionary, modified as necessary
    :param iteration: Iteration number to be compared to
                the 'first_selfcal' field.
    :param tol: Iteration stops when the fractional change
                 in the gain solution is below this tolerance
    :return: Calibrated data_models, dict(GainTables)
    """
    if controls is None:
        controls = create_calibration_controls()

    avis = vis
    amvis = model_vis

    if isinstance(gaintables, GainTable):
        gaintables = [gaintables]

    gt = {}
    # Use the existing gaintables if needed
    if isinstance(gaintables, list):
        for gaintable in gaintables:
            if gaintable.attrs["jones_type"] in list(calibration_context):
                gt[gaintable.attrs["jones_type"]] = gaintable
    elif isinstance(gaintables, dict):
        gt = gaintables

    for c in list(calibration_context):
        if iteration >= controls[c]["first_selfcal"]:
            if c not in gt.keys():
                log.info("Creating new %s gaintable", c)
                gt[c] = create_gaintable_from_visibility(
                    avis, timeslice=controls[c]["timeslice"], jones_type=c
                )

            gt[c] = solve_gaintable(
                avis,
                amvis,
                gain_table=gt[c],
                phase_only=controls[c]["phase_only"],
                crosspol=controls[c]["shape"] == "matrix",
                timeslice=controls[c]["timeslice"],
                tol=tol,
            )
            log.debug(
                "calibrate_chain: Jones matrix %s, iteration %s",
                c,
                iteration,
            )
            log.debug(
                gt[c].gaintable_acc.qa_gain_table(
                    context=f"Jones matrix {c}, iteration {iteration}"
                )
            )
            avis = apply_gaintable(
                avis,
                gt[c],
                inverse=True,
            )
        else:
            log.debug(
                "calibrate_chain: Jones matrix %s, iteration %s",
                c,
                iteration,
            )

    return avis, gt


def solve_calibrate_chain(
    vis,
    model_vis,
    gaintables=None,
    calibration_context="T",
    controls=None,
    iteration=0,
    tol=1e-6,
):
    """
    Solve GainTables by fitting an observed visibility
    to a model visibility.

    The context string can denote a sequence of calibrations
    e.g. TGB.
    Currently, we do not support inputting different timescales.

    :param vis: Visibility containing the observed data_models
    :param model_vis: Visibility containing the visibility predicted by a model
    :param gaintables: Existing GainTables (GainTable, list or dict)
    :param calibration_context: calibration contexts in order
                    of correction e.g. 'TGB'
    :param controls: controls dictionary, modified as necessary
    :param iteration: Iteration number to be compared to
                    the 'first_selfcal' field.
    :param tol: Iteration stops when the fractional change
                 in the gain solution is below this tolerance
    :return: dict(GainTables)
    """
    if controls is None:
        controls = create_calibration_controls()

    avis = vis
    amvis = model_vis

    if isinstance(gaintables, GainTable):
        gaintables = [gaintables]

    gt = {}
    # Use the existing gaintables if needed
    if isinstance(gaintables, list):
        for gaintable in gaintables:
            if gaintable.attrs["jones_type"] in list(calibration_context):
                gt[gaintable.attrs["jones_type"]] = gaintable
    elif isinstance(gaintables, dict):
        gt = gaintables

    for c in list(calibration_context):
        if c not in gt.keys():
            gt[c] = create_gaintable_from_visibility(
                avis, timeslice=controls[c]["timeslice"], jones_type=c
            )
        fmin = gt[c].frequency.data[0]
        fmax = gt[c].frequency.data[-1]
        if iteration >= controls[c]["first_selfcal"]:
            if numpy.max(
                numpy.abs(vis.visibility_acc.flagged_weight)
            ) > 0.0 and (
                amvis is None or numpy.max(numpy.abs(amvis.vis)) > 0.0
            ):
                gt[c] = solve_gaintable(
                    avis,
                    amvis,
                    gain_table=gt[c],
                    phase_only=controls[c]["phase_only"],
                    crosspol=controls[c]["shape"] == "matrix",
                    timeslice=controls[c]["timeslice"],
                    tol=tol,
                )
                context_message = (
                    f"Model is non-zero: solving for Jones matrix {c}, "
                    f"iteration {iteration}, frequency "
                    f"{fmin:4g} - {fmax:4g} Hz"
                )
                qa = gt[c].gaintable_acc.qa_gain_table(context=context_message)
                log.info("calibrate_chain: %s", qa)
            else:
                log.info(
                    "No model data: cannot solve for Jones matrix %s, "
                    "iteration %s, frequency %4g - %4g Hz",
                    c,
                    iteration,
                    fmin,
                    fmax,
                )
        else:
            log.info(
                "Not solving for Jones matrix %s this iteration: "
                "iteration %s, frequency %4g - %4g Hz",
                c,
                iteration,
                fmin,
                fmax,
            )

    return gt
