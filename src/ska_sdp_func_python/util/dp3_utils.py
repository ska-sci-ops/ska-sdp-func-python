"""
Functions to use DP3 steps with visibility data class from ska_sdp_datamodels.
"""

__all__ = [
    "create_dp_info",
    "process_visibilities",
]

import numpy as np

from ska_sdp_func_python.visibility.operations import (
    copy_data_and_shrink_polarizations,
    expand_polarizations,
)


def create_dp_info(vis):
    """Creates a DPInfo object based on input visibilities.

    :param vis: Visibility object (or graph)
    :return: DPInfo to use for DP3 steps
    """

    import dp3  # noqa: E501 # pylint:disable=no-name-in-module,import-error,import-outside-toplevel

    # Some DP3 steps only work with 4 correlations (gaincal, for example)
    # hence we force the number of correlations to 4 in the DPInfo, and do a
    # mapping from 1,2 correlations to 4 when needed (in the
    # process_visibilities function)
    n_correlations = 4
    dpinfo = dp3.DPInfo(n_correlations)
    dpinfo.set_channels(vis.frequency.data, vis.channel_bandwidth.data)

    antenna1 = vis.antenna1.data
    antenna2 = vis.antenna2.data
    antenna_names = vis.configuration.names.data
    antenna_positions = vis.configuration.xyz.data
    antenna_diameters = vis.configuration.diameter.data
    dpinfo.set_antennas(
        antenna_names,
        antenna_diameters,
        antenna_positions,
        antenna1,
        antenna2,
    )
    first_time = vis.time.data[0]
    last_time = vis.time.data[-1]
    time_interval = vis.integration_time.data[0]
    dpinfo.set_times(first_time, last_time, time_interval)
    dpinfo.phase_center = [vis.phasecentre.ra.rad, vis.phasecentre.dec.rad]

    return dpinfo


def process_visibilities(
    step, vis, *, save_out_vis=True, extra_data_name=None, extra_data=None
):
    """Process visibilities with the operation defined by the input step.

    :param step: DP3 step
    :param vis: Visibility object (or graph)
    :param save_out_vis: set to False if the output visibilities are not
                        needed (for example if the step should produce
                        solutions, which are saved to disk)
    :param extra_data_name: Name of the extra visibility data the step should
                            use (for example model data)
    :param extra_data: Visibility object (or graph) containing the extra data
    :return: Processed visibilities
    """

    import dp3  # noqa: E501 # pylint:disable=no-name-in-module,import-error,import-outside-toplevel
    import dp3.steps  # noqa: E501 # pylint:disable=no-name-in-module,import-error,import-outside-toplevel
    from dp3.parameterset import (  # noqa: E501 # pylint:disable=no-name-in-module,import-error,import-outside-toplevel
        ParameterSet,
    )

    # To extract the visibilities processed by the step, we add a QueueOutput
    # step which will accumulate the output visibilities.
    if save_out_vis:
        queue_step = dp3.steps.QueueOutput(ParameterSet(), "")
        step.set_next_step(queue_step)
    else:
        step.set_next_step(
            dp3.make_step("null", ParameterSet(), "", dp3.MsType.regular)
        )

    step.set_info(create_dp_info(vis))

    # The time_index variable is needed to extract the correct time from the
    # "extra_data" visibility object, which is not indexed in the for loop
    time_index = 0
    for time, vis_per_timeslot in vis.groupby("time"):
        dpbuffer = dp3.DPBuffer()
        dpbuffer.set_data(
            expand_polarizations(vis_per_timeslot.vis.data, np.complex64)
        )
        dpbuffer.set_weights(
            expand_polarizations(vis_per_timeslot.weight.data, np.float32)
        )
        dpbuffer.set_flags(
            expand_polarizations(vis_per_timeslot.flags.data, bool)
        )
        dpbuffer.set_uvw(-vis_per_timeslot.uvw.data)
        if extra_data_name:
            time_slice = extra_data.time.data[time_index]
            model_vis_per_timeslot = extra_data.sel({"time": time_slice})
            modeldata = expand_polarizations(
                model_vis_per_timeslot.vis.data, np.complex64
            )
            dpbuffer.add_data(extra_data_name)
            dpbuffer.set_extra_data(extra_data_name, modeldata)
        time_index += 1

        dpbuffer.set_time(time)
        step.process(dpbuffer)
    step.finish()

    # Extract the results from the QueueOutput step, if desired.
    if save_out_vis:

        time_index = 0

        while not queue_step.queue.empty():
            dpbuffer_from_queue = queue_step.queue.get()
            visibilities_out = np.array(
                dpbuffer_from_queue.get_data(), copy=False
            )
            flags_out = np.array(dpbuffer_from_queue.get_flags(), copy=False)
            weights_out = np.array(
                dpbuffer_from_queue.get_weights(), copy=False
            )
            uvws_out = np.array(dpbuffer_from_queue.get_uvw(), copy=False)

            nr_polarizations = vis.vis.data.shape[-1]
            vis.vis.data[
                time_index, :, :, :
            ] = copy_data_and_shrink_polarizations(
                visibilities_out, nr_polarizations
            )
            vis.flags.data[
                time_index, :, :, :
            ] = copy_data_and_shrink_polarizations(flags_out, nr_polarizations)
            vis.weight.data[
                time_index, :, :, :
            ] = copy_data_and_shrink_polarizations(
                weights_out, nr_polarizations
            )

            vis.uvw.data[time_index, :, :] = uvws_out

            time_index += 1

        return vis

    return None
