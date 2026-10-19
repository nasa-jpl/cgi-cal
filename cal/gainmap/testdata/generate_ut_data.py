# Copyright 2025, by the California Institute of Technology.
# ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.
# Any commercial use must be negotiated with the Office of Technology Transfer
# at the California Institute of Technology.
import os
import numpy as np
from astropy.io import fits

nActs = 48
# quadratic model of height as a function of command voltage
a = 1/100
b = 3
h0 = 0

# command voltages for height slices
hvolts = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
# values for each slice (assumes all actuator values of each slice share the
# same value)
hslicevals = a*hvolts**2 + b*hvolts + h0

#initializing
hcube = np.ones((len(hvolts), nActs, nActs))

# assigning values
for i in range(len(hcube)):
    hcube[i] *= hslicevals[i]

localpath = os.path.dirname(os.path.abspath(__file__))

# saving height cube
hdr = fits.Header()
prim = fits.PrimaryHDU(header=hdr)
img = fits.ImageHDU(hcube)
hdul = fits.HDUList([prim, img])
hdul.writeto(os.path.join(localpath, 'ut_refCubeQuadratic.fits'),
             overwrite=True)

# now for command voltages for height cube
img = fits.ImageHDU(hvolts)
hdul = fits.HDUList([prim, img])
hdul.writeto(os.path.join(localpath, 'ut_refCommandVec.fits'), overwrite=True)

# making gain cube and corresponding command volatge vector
gvolts = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90])
# values for each slice (assumes all actuator values of each slice share the
# same value)
# g(v) = dh(v)/dv:
gslicevals = 2*a*gvolts + b

# initializing
gcube = np.ones((len(gvolts), nActs, nActs))

# assigning values
for i in range(len(gcube)):
    gcube[i] *= gslicevals[i]

# saving gain cube
hdr = fits.Header()
prim = fits.PrimaryHDU(header=hdr)
img = fits.ImageHDU(gcube)
hdul = fits.HDUList([prim, img])
hdul.writeto(os.path.join(localpath, 'ut_refCubeQuadratic_Gain.fits'),
             overwrite=True)

# now for command voltages for gain cube
img = fits.ImageHDU(gvolts)
hdul = fits.HDUList([prim, img])
hdul.writeto(os.path.join(localpath, 'ut_refCommandVec_Gain.fits'),
             overwrite=True)

# height slice values for negative-gain 
zero = 3
b2 = -1
const = 30
hslicevals = (hvolts-zero)**3 + b2*hvolts**2 + const

#initializing
hcube = np.ones((len(hvolts), nActs, nActs))

# assigning values
for i in range(len(hcube)):
    hcube[i] *= hslicevals[i]

# saving height cube
hdr = fits.Header()
prim = fits.PrimaryHDU(header=hdr)
img = fits.ImageHDU(hcube)
hdul = fits.HDUList([prim, img])
hdul.writeto(os.path.join(localpath, 'ut_refCubeNegGain.fits'),
             overwrite=True)

# corresponding gain cube
# g(v) = dh(v)/dv:
gslicevals = 3*(gvolts-zero)**2 +2*b2*gvolts

# initializing
gcube = np.ones((len(gvolts), nActs, nActs))

# assigning values
for i in range(len(gcube)):
    gcube[i] *= gslicevals[i]

# saving gain cube
hdr = fits.Header()
prim = fits.PrimaryHDU(header=hdr)
img = fits.ImageHDU(gcube)
hdul = fits.HDUList([prim, img])
hdul.writeto(os.path.join(localpath, 'ut_refCubeNegGain_Gain.fits'),
             overwrite=True)