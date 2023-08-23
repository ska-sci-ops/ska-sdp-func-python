.. _dp3:

DP3 support
===

This repository offers the possibility to use 
`DP3 <https://dp3.readthedocs.io/en/latest/>`_ steps with a  
`Visibility <https://developer.skao.int/projects/ska-sdp-datamodels/en/latest/api/ska_sdp_datamodels.visibility.Visibility.html>`_ 
object as input/output.

Any step can be used via the function 'process_visibilities', where the desired 
step is given via the 'step' parameter. Wrappers are defined for the DP3 
Predict and Gaincal steps.

Some examples are reported in the  :ref:`example <_usage>`: section.

.. toctree::
  
   ska_sdp_func_api/dp3_support/index.rst

Indexes
+++++++

* :ref:`genindex`
* :ref:`modindex`