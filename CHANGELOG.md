# Changelog

* Add functions process_visibilities and create_dpinfo to ease the use of DP3 steps ([MR38](https://gitlab.com/ska-telescope/sdp/ska-sdp-func-python/-/merge_requests/38))

0.2.4
----
* Update xarray to newest version and resolved dependencies ([MR37](https://gitlab.com/ska-telescope/sdp/ska-sdp-func-python/-/merge_requests/37))

0.2.3
----
* Added searching best reference antenna for gain calibration ([MR36](https://gitlab.com/ska-telescope/sdp/ska-sdp-func-python/-/merge_requests/36))

0.2.2
----
* Fixing partial gain solver errors ([MR34](https://gitlab.com/ska-telescope/sdp/ska-sdp-func-python/-/merge_requests/34)) 

0.2.1
----
* Fixing function inputs in chain_calibration.py ([MR31](https://gitlab.com/ska-telescope/sdp/ska-sdp-func-python/-/merge_requests/31))
* Added utility functions to support rechannelisation of bandpass and delay solutions for CBF beamformer calibration ([MR29](https://gitlab.com/ska-telescope/sdp/ska-sdp-func-python/-/merge_requests/29))

0.2.0
----
* Fix mapping of calibration techniques between rascil and DP3, add more extensive tests ([MR30](https://gitlab.com/ska-telescope/sdp/ska-sdp-func-python/-/merge_requests/30))

0.1.5
----
* Bug fix for calculating f2 of multi-channels ([MR26](https://gitlab.com/ska-telescope/sdp/ska-sdp-func-python/-/merge_requests/26))

0.1.4
----
* Migrate function to perform DP3 calibration from rascil-main ([MR24](https://gitlab.com/ska-telescope/sdp/ska-sdp-func-python/-/merge_requests/24))
* Bug fix for weight_visibility function modified input visibility ([MR23](https://gitlab.com/ska-telescope/sdp/ska-sdp-func-python/-/merge_requests/23))

0.1.3
----
* Bug fix for calculating weighing with conjugate visibilities ([MR20](https://gitlab.com/ska-telescope/sdp/ska-sdp-func-python/-/merge_requests/20))
* Add function expand_polarizations, needed to add the option of calibration with DP3 in rascil ([MR22](https://gitlab.com/ska-telescope/sdp/ska-sdp-func-python/-/merge_requests/22))

0.1.2
----
* Refactored various functions related to deconvolution and calibration to decrease complexity ([MR12](https://gitlab.com/ska-telescope/sdp/ska-sdp-func-python/-/merge_requests/12))
* Fixed some bugs and tests for functions with image creation, apply_gaintable and advise_wide_field ([MR11](https://gitlab.com/ska-telescope/sdp/ska-sdp-func-python/-/merge_requests/11), [MR14](https://gitlab.com/ska-telescope/sdp/ska-sdp-func-python/-/merge_requests/14), [MR15](https://gitlab.com/ska-telescope/sdp/ska-sdp-func-python/-/merge_requests/15))
* Reviewed and updated processing functions documentation ([MR8](https://gitlab.com/ska-telescope/sdp/ska-sdp-func-python/-/merge_requests/8), [MR13](https://gitlab.com/ska-telescope/sdp/ska-sdp-func-python/-/merge_requests/13))
* Rearranged files and renamed directories ([MR4](https://gitlab.com/ska-telescope/sdp/ska-sdp-func-python/-/merge_requests/4), [MR5](https://gitlab.com/ska-telescope/sdp/ska-sdp-func-python/-/merge_requests/5), [MR7](https://gitlab.com/ska-telescope/sdp/ska-sdp-func-python/-/merge_requests/7))
* Migrated processing functions from RASCIL ([MR2](https://gitlab.com/ska-telescope/sdp/ska-sdp-func-python/-/merge_requests/2), [MR6](https://gitlab.com/ska-telescope/sdp/ska-sdp-func-python/-/merge_requests/6))
