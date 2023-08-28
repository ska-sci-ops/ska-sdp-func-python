.. _documentation_master:

.. toctree::

SKA SDP Python Processing Functions
###################################
This `repository <https://gitlab.com/ska-telescope/sdp/ska-sdp-func-python.git>`_
contains Processing Function wrappers implemented in Python. The original code was migrated
from `RASCIL <https://gitlab.com/ska-telescope/external/rascil-main.git>`_. They provide reference
implementations for lower-level processing function libraries, such as
`ska-sdp-func <https://gitlab.com/ska-telescope/sdp/ska-sdp-func.git>`_
and an interface between these and high-level data models
(`ska-sdp-datamodel <https://gitlab.com/ska-telescope/sdp/ska-sdp-datamodels.git>`_).


Installation Instructions
=========================

The package is installable via pip::

    pip install ska-sdp-func-python --extra-index-url https://artefact.skao.int/repository/pypi-internal/simple

If you would like to view the source code or install from git, use::

    git clone https://gitlab.com/ska-telescope/sdp/ska-sdp-func-python.git

Please ensure you have all the dependency packages installed. The installation is managed
through `poetry <https://python-poetry.org/docs/>`_.
Refer to their page for instructions.


.. toctree::
   :maxdepth: 1
   :caption: Sections

   fourier_processing
   ionospheric_calibration
   functions
   usage
   api
   dp3




