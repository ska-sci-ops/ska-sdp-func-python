.. _usage:

.. toctree::
  :maxdepth: 2

Usage Examples
==============

Calibration
-----------

Calibration control is via a calibration_controls dictionary
created by :py:func:`ska_sdp_func_python.calibration.chain_calibration.create_calibration_controls`.
This supports the following Jones matrices::

   . T - Atmospheric phase
   . G - Electronics gain
   . P - Polarisation
   . B - Bandpass
   . I - Ionosphere

This is specified via a dictionary::

    contexts = {'T': {'shape': 'scalar', 'timeslice': 'auto',
                        'phase_only': True, 'first_iteration': 0},
                'G': {'shape': 'vector', 'timeslice': 60.0,
                        'phase_only': False, 'first_iteration': 0},
                'P': {'shape': 'matrix', 'timeslice': 1e4,
                        'phase_only': False, 'first_iteration': 0},
                'B': {'shape': 'vector', 'timeslice': 1e5,
                        'phase_only': False, 'first_iteration': 0},
                'I': {'shape': 'vector', 'timeslice': 1.0,
                        'phase_only': True, 'first_iteration': 0}}

Currently P and I are not supported.

For example::

    controls = create_calibration_controls()

    controls['T']['first_selfcal'] = 1
    controls['T']['phase_only'] = True
    controls['T']['timeslice'] = 'auto'

    controls['G']['first_selfcal'] = 3
    controls['G']['timeslice'] = 'auto'

    controls['B']['first_selfcal'] = 4
    controls['B']['timeslice'] = 1e5

    ical_list = ical_list_rsexecute_workflow(vis_list,
                                              model_imagelist=future_model_list,
                                              context='wstack', vis_slices=51,
                                              scales=[0, 3, 10], algorithm='mmclean',
                                              nmoment=3, niter=1000,
                                              fractional_threshold=0.1,
                                              threshold=0.1, nmajor=5, gain=0.25,
                                              deconvolve_facets=1,
                                              deconvolve_overlap=0,
                                              deconvolve_taper='tukey',
                                              timeslice='auto',
                                              psf_support=64,
                                              global_solution=False,
                                              calibration_context='TGB',
                                              do_selfcal=True)


Calibration solvers are via substitution algorithm due to
Larry D'Addario c 1980'ish. Used in the original VLA Dec-10 Antsol.

For example::

    gtsol = solve_gaintable(vis, originalvis,
            phase_only=True, niter=niter, crosspol=False, tol=1e-6)
    vis = apply_gaintable(vis, gtsol, inverse=True)

It is also possible to run  `DP3 Gaincal step <https://dp3.readthedocs.io/>` 
with :py:func:`sks_sdp_func_python.calibration.dp3_calibration.dp3_gaincal`:
The skymodel needs to be converted in a text format prior to the calibration.

For example::

    export_skymodel_to_text(SkyModel(sky_components), "dp3.skymodel")
    dp3_gaincal(visibility, ["T"], True, "dp3.skymodel")

Fourier Transforms
------------------

All grids and images are considered quadratic and centered around
`npixel//2`, where `npixel` is the pixel width/height.
This means that `npixel//2` is the zero frequency for FFT purposes,
as is convention. Note that this means that for even `npixel` the
grid is not symmetrical, which means that e.g. for convolution
kernels odd image sizes are preferred.

Gridding
--------

Imaging is based on use of the FFT to perform Fourier transforms
efficiently. Since the observed visibility data_models
do not arrive naturally on grid points, the sampled points are
resampled on the FFT grid using a convolution function to
smear out the sample points. The resulting grid points are then FFT'ed.
The result can be corrected for the griddata convolution function by
division in the image plane of the transform.

The grid_data module :py:func:`ska_sdp_func_python.grid_data` contains functions for performing the
griddata process and the inverse degridding process.

GridData, ConvolutionFunction and Vis
always have the same PolarisationFrame. Conversion to
stokesIQUV is only done in the image plane. These data models can be found in
`ska-sdp-datamodel <https://gitlab.com/ska-telescope/sdp/ska-sdp-datamodels.git>`

Image
-----

Functions in the Image module :py:func:`ska_sdp_func_python.image` aid Fourier transform processing.
These are built on top of the core functions in
:py:func:`ska_sdp_func_python.fourier_transforms`.

The measurement equation for a sufficiently narrow
field of view interferometer is:

.. math::

    V(u,v,w) =int I(l,m) e^{-2 pi j (ul+vm)} dl dm


The measurement equation for a wide field of view interferometer is:

 .. math::

     V(u,v,w) = int frac{I(l,m)}{sqrt{1-l^2-m^2}}
         e^{-2 pi j (ul+vm + w(sqrt{1-l^2-m^2}-1))} dl dm

This and related modules contain various approaches for dealing with
the wide-field problem where the extra phase term in the Fourier
transform cannot be ignored.

The standard deconvolution algorithms are provided by
:py:func:`ska_sdp_func_python.imaging.cleaners`.

   * hogbom: Hogbom CLEAN See: Hogbom CLEAN A&A Suppl, 15, 417, (1974)

   * msclean: MultiScale CLEAN See: Cornwell, T.J., Multiscale CLEAN
     (IEEE Journal of Selected Topics in Sig Proc,
     2008 vol. 2 pp. 793-801)

   * mfsmsclean: MultiScale Multi-Frequency
     See: U. Rau and T. J. Cornwell, “A multi-scale multi-frequency
     deconvolution algorithm for synthesis imaging in radio interferometry,”
     A&A 532, A71 (2011).

For example to make dirty image and PSF, deconvolve, and then restore::

    model = create_image_from_visibility(vt, cellsize=0.001, npixel=256)
    dirty, sumwt = invert_visibility(vt, model, context="2d")
    psf, sumwt = invert_visibility(vt, model, context="2d", dopsf=True)

    comp, residual = deconvolve_cube(dirty, psf, niter=1000, threshold=0.001,
                        fracthresh=0.01, window_shape='quarter',
                        gain=0.7, algorithm='msclean', scales=[0, 3, 10, 30])

    restored = restore_cube(comp, psf, residual)

All functions return an image holding clean components and residual image.

Imaging
-------

The imaging functions in :py:func:`ska_sdp_func_python.imaging` include 2D prediction
and inversion operations. A very simple example, given a model Image to specify the
image size, sampling, and phasecentre::

    model = create_image_from_visibility(vis, npixel=1024, nchan=1)
    dirty, sumwt = invert_visibility(vis, model, context="2d")

The call to create_image_from_visibility step constructs a template image.
The dirty image is constructed according to this template.

AW projection is supported by the predict_visibility and invert_visibility methods,
provided the gridding kernel is constructed and passed in as a partial.
For example::

    gcfcf = functools.partial(create_awterm_convolutionfunction, nw=100, wstep=8.0,
            oversampling=8, support=100, use_aaf=True)
    dirty, sumwt = invert_visibility(vis, model, context="awprojection", gcfcf=gcfcf)

If installed, the nifty gridder (https://gitlab.mpcdf.mpg.de/ift/nifty_gridder)
can also be used::

    dirty, sumwt = invert_visibility(vis, model, verbosity=2, context="ng")

The convolutional gridding functions are to be found in the grid_data module

.. _dp3_usage:

DP3
---
In order to use DP3, the optional dp3 dependency must be installed. You can obtain it with::

    pip install dp3

The functions in the module :py:func:`ska_sdp_func_python.util.dp3_utils` allow 
usage of DP3 steps. The user should define the parset for the specific step 
with the desired settings, and use the parset to create the step using the 
function :py:func:`dp3.make_step`. ::

    import dp3
    parset = dp3.parameterset.ParameterSet()
    parset.add("predict.sourcedb", "test.skymodel")
    predict_step = dp3.make_step("predict", parset, "predict.", dp3.MsType.regular)

Once the step object is created, the visibilities can be passed to it and 
processed with the :py:func:`process_visibilities` function ::

    from ska_sdp_func_python.util.dp3_utils import process_visibilities
    predicted_vis = process_visibilities(predict_step, input_visibilities)

For some steps, it is possible to call the function directly. This is the case 
for `Predict <https://dp3.readthedocs.io/en/latest/steps/Predict.html>`_  and 
`Gaincal <https://dp3.readthedocs.io/en/latest/steps/GainCal.html>`_ :: 

    from ska_sdp_func_python.calibration.dp3_calibration import dp3_gaincal
    calibrated_vis = dp3_gaincal(skymodel_vis, calibration_context, global_solution, solutions_filename)