"""
Functions to use DP3 for calibration purposes.
"""

__all__ = [
    "dp3_gaincal",
    "dp3_gaincal_with_modeldata",
]

import logging

import numpy

from ska_sdp_func_python.calibration.chain_calibration import (
    create_calibration_controls,
)

log = logging.getLogger("func-python-logger")


def create_parset_from_context(
    vis,
    calibration_context,
    global_solution,
    solutions_filename,
    *,
    skymodel_filename=None,
    apply_solutions=True,
    modeldata_name=None,
):
    """Defines input parset for DP3 based on calibration context.

    :param vis: Visibility object
    :param calibration_context: String giving terms to be calibrated e.g. 'TGB'
    :param global_solution: Find a single solution over all frequency channels
    :param solutions_filename: Filename of the calibration solutions produced
    by DP3. The easiest way to inspect the results is by using a .h5 extension.
    :param skymodel_filename: Filename of the skymodel used by DP3
    :param apply_solutions: Apply the calibration solution to the visibility
    :param modeldata_name: Name of the modeldata as specificed in the DPBuffer.
    If this is given, the skymodel_filename won't be read
    :return: list of parsets for the different calibrations to run
    """

    from dp3.parameterset import (  # noqa: E501 # pylint: disable=import-error,import-outside-toplevel
        ParameterSet,
    )

    parset_list = []
    controls = create_calibration_controls()
    for calibration_control in calibration_context:
        parset = ParameterSet()

        parset.add("gaincal.parmdb", solutions_filename)
        if modeldata_name:
            parset.add("gaincal.reusemodel", modeldata_name)
        else:
            parset.add("gaincal.sourcedb", skymodel_filename)
        timeslice = controls[calibration_control]["timeslice"]
        if timeslice == "auto" or timeslice is None or timeslice <= 0.0:
            parset.add("gaincal.solint", "1")
        else:
            nbins = max(
                1,
                numpy.ceil(
                    (numpy.max(vis.time.data) - numpy.min(vis.time.data))
                    / timeslice
                ).astype("int"),
            )
            parset.add("gaincal.solint", str(nbins))
        if global_solution:
            parset.add("gaincal.nchan", "0")
        else:
            parset.add("gaincal.nchan", "1")

        if apply_solutions:
            parset.add("gaincal.applysolution", "true")

        if controls[calibration_control]["phase_only"]:
            if controls[calibration_control]["shape"] == "vector":
                parset.add("gaincal.caltype", "diagonalphase")
            elif controls[calibration_control]["shape"] == "matrix":
                parset.add("gaincal.caltype", "fulljones")
            else:
                parset.add("gaincal.caltype", "scalarphase")
        else:
            if controls[calibration_control]["shape"] == "vector":
                parset.add("gaincal.caltype", "diagonal")
            elif controls[calibration_control]["shape"] == "matrix":
                parset.add("gaincal.caltype", "fulljones")
            else:
                parset.add("gaincal.caltype", "scalar")

        parset_list.append(parset)

    return parset_list


def dp3_gaincal(
    vis,
    calibration_context,
    global_solution,
    skymodel_filename="test.skymodel",
    solutions_filename="gaincal.h5",
):
    """Calibrates visibilities using the DP3 package.

    :param vis: Visibility object (or graph)
    :param calibration_context: String giving terms to be calibrated e.g. 'TGB'
    :param global_solution: Solve for global gains
    :param skymodel_filename: Filename of the skymodel used by DP3
    :return: calibrated visibilities
    """
    from dp3 import (  # noqa: E501 # pylint:disable=no-name-in-module,import-error,import-outside-toplevel
        MsType,
        make_step,
    )

    from ska_sdp_func_python.util.dp3_utils import process_visibilities

    log.info("Started computing dp3_gaincal")
    calibrated_vis = vis.copy(deep=True)

    parset_list = create_parset_from_context(
        calibrated_vis,
        calibration_context,
        global_solution,
        solutions_filename,
        skymodel_filename=skymodel_filename,
    )

    for parset in parset_list:
        gaincal_step = make_step(
            "gaincal",
            parset,
            "gaincal.",
            MsType.regular,
        )

        process_visibilities(gaincal_step, calibrated_vis)

        log.info("Finished computing dp3_gaincal")

    return calibrated_vis


def dp3_gaincal_with_modeldata(
    vis,
    calibration_context,
    global_solution,
    model_vis,
    modeldata_name,
    solutions_filename="gaincal.h5",
):
    """Calibrates visibilities using the DP3 package, with model data provided
       as a separate visibility object.

    :param vis: Visibility object (or graph)
    :param calibration_context: String giving terms to be calibrated e.g. 'TGB'
    :param global_solution: Solve for global gains
    :param model_vis: Visibility object (or graph) containing the extra data
    :param modeldata_name: Name of the model visibility data the step should
                           use (for example "modeldata")
    :return: calibrated visibilities
    """

    from dp3 import (  # noqa: E501 # pylint:disable=no-name-in-module,import-error,import-outside-toplevel
        MsType,
        make_step,
    )

    from ska_sdp_func_python.util.dp3_utils import process_visibilities

    log.info("Started computing dp3_gaincal_with_modeldata")
    calibrated_vis = vis.copy(deep=True)

    parset_list = create_parset_from_context(
        calibrated_vis,
        calibration_context,
        global_solution,
        solutions_filename,
        modeldata_name=modeldata_name,
    )

    for parset in parset_list:
        gaincal_step = make_step(
            "gaincal",
            parset,
            "gaincal.",
            MsType.regular,
        )
        process_visibilities(
            gaincal_step,
            calibrated_vis,
            extra_data_name=modeldata_name,
            extra_data=model_vis,
        )

    log.info("Finished computing dp3_gaincal_with_modeldata")

    return calibrated_vis
