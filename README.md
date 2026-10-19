# Cal
Calibration tools for WFIRST-CGI.  Each folder within `cal` is a
different tool.  See individual README files for more details.

## Installing
To install simply download the package, change directories into the downloaded folder, and run:

	pip install .

Calibration requires the following packages to be installed for use:

* astropy
* matplotlib
* numpy
* opencv-python
* pandas
* pyyaml
* scikit-image
* scipy

For some scripts to run, certain packages that are not included with the installation may be needed, such as:
cgisim:  [https://sourceforge.net/projects/cgisim/](https://sourceforge.net/projects/cgisim/)
HOWFSC:  [https://github.jpl.nasa.gov/WFIRST-CGI/HOWFS](https://github.jpl.nasa.gov/WFIRST-CGI/HOWFS)
emccd_detect:  [https://github.jpl.nasa.gov/WFIRST-CGI/emccd_detect](https://github.jpl.nasa.gov/WFIRST-CGI/emccd_detect)
FALCO:  [https://github.com/ajeldorado/falco-python](https://github.com/ajeldorado/falco-python)
PROPER:  [http://proper-library.sourceforge.net](http://proper-library.sourceforge.net)

## Copyright statement
Copyright 2025, by the California Institute of Technology. ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged. Any commercial use must be negotiated with the Office of Technology Transfer at the California Institute of Technology.

## Authors

* AJ Riggs
* David Marx
* Eric Cady
* Kevin Ludwick
* Sam Halverson
