# Purpose of AMPTHRESH

From the __WFIRST Coronagraph Instrument Operations Concept Document__:

Given an amplitude or intensity measurement on a pupil-plane image, threshold it to select pixels which are inside the open area. Used to create masks to eliminate low SNR regions.

The AMPTHRESH algorithm needs a standardized input format for images or phase retrieval data.

# Running Tests
To run tests for AMPTHRESH, type this in the command line in the ampthresh folder (replace "python" with "python3" if "python" defaults to python2:

__python testsuite.py__


# Contents of This Folder

1. README.md : This file

2. ampthresh.py : File containing the ampthresh algorithm and directly supporting functions.

3. testdata : Folder containing the FITS files used for unit and integration tests.

4. testsuite.py : A simple python script used for testing. Looks for and runs using unittest all files named in the format "ut_*.py".

5. ut_ampthresh.py : File containing all the unit and integration tests for AMPTHRESH.

6. util/check.py : Functions for checking inputs to other functions.

7. old : Folder containing some earlier versions of scripts and functions.
